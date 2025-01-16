import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union

from .modules import (
    ScalarEncoder,
    EntityEncoder,
    SpatialEncoder,
    AlphaStarCore,
    ActionHeads
)

class AlphaStar(nn.Module):
    """
    Complete AlphaStar architecture combining supervised learning and reinforcement learning components.
    
    The model consists of:
    1. Feature extractors for different input modalities
    2. LSTM core for temporal processing
    3. Auto-regressive policy heads for action selection
    4. Value heads for both supervised and RL training
    5. Auxiliary prediction heads for enhanced learning
    
    Key Components:
    - Scalar features: Game statistics, resources, etc.
    - Entity features: Units, buildings, etc. processed by transformer
    - Spatial features: Map information processed by ResNet
    - LSTM core: Temporal dependencies and memory
    - Action heads: Auto-regressive policy for complex action space
    - Value heads: State-value estimation and baseline functions
    
    Training Modes:
    - Supervised Learning: Learning from human demonstrations
    - Reinforcement Learning: Self-play and league training
    - Auxiliary Tasks: Additional supervision signals
    """
    
    def __init__(self, config: Dict):
        """
        Initialize AlphaStar model.
        
        Args:
            config: Configuration dictionary containing:
                - input_dims: Dimensions for different input modalities
                - network_params: Architecture parameters (hidden dims, layers, etc.)
                - action_space: Action space configuration
                - training_params: Parameters for different training modes
        """
        super().__init__()
        
        # Feature encoders
        self.scalar_encoder = ScalarEncoder(
            input_dim=config['input_dims']['scalar'],
            hidden_dims=config['network_params']['scalar_encoder'],
            output_dim=config['network_params']['encoder_output']
        )
        
        self.entity_encoder = EntityEncoder(
            input_dim=config['input_dims']['entity'],
            d_model=config['network_params']['transformer_dim'],
            num_heads=config['network_params']['transformer_heads'],
            num_layers=config['network_params']['transformer_layers'],
            dropout=config['network_params']['dropout']
        )
        
        self.spatial_encoder = SpatialEncoder(
            input_channels=config['input_dims']['spatial'],
            base_channels=config['network_params']['spatial_channels'],
            num_res_blocks=config['network_params']['num_res_blocks']
        )
        
        # Calculate total encoded dimension
        self.encoded_dim = (
            config['network_params']['encoder_output'] +  # Scalar
            config['network_params']['transformer_dim'] +  # Entity
            config['network_params']['spatial_channels']   # Spatial
        )
        
        # Core LSTM
        self.core = AlphaStarCore(
            input_dim=self.encoded_dim,
            hidden_dim=config['network_params']['lstm_dim'],
            num_layers=config['network_params']['lstm_layers'],
            dropout=config['network_params']['dropout']
        )
        
        # Action heads for auto-regressive policy
        self.action_heads = ActionHeads(
            core_dim=config['network_params']['lstm_dim'],
            entity_dim=config['input_dims']['entity'],
            num_actions=config['action_space']['num_actions'],
            spatial_size=config['action_space']['spatial_size']
        )
        
        # Value heads
        self.supervised_value = nn.Sequential(
            nn.Linear(config['network_params']['lstm_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.baseline_value = nn.Sequential(
            nn.Linear(config['network_params']['lstm_dim'], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Auxiliary prediction heads
        if config['training_params'].get('use_auxiliary', False):
            self.build_auxiliary_heads(config)
        
        self.config = config
    
    def build_auxiliary_heads(self, config: Dict):
        """Build auxiliary prediction heads for additional supervision."""
        self.auxiliary_heads = nn.ModuleDict({
            'build_order': nn.Linear(
                config['network_params']['lstm_dim'],
                config['action_space']['build_order_size']
            ),
            'unit_counts': nn.Linear(
                config['network_params']['lstm_dim'],
                config['action_space']['unit_type_size']
            )
        })
    
    def encode_inputs(self, 
                     inputs: Dict[str, torch.Tensor],
                     masks: Optional[Dict[str, torch.Tensor]] = None
                    ) -> torch.Tensor:
        """
        Encode different input modalities into a unified representation.
        
        Args:
            inputs: Dictionary containing different input modalities
            masks: Optional masks for different input types
            
        Returns:
            Tensor of shape (batch_size, encoded_dim) containing unified representation
        """
        # Encode different input streams
        scalar_encoded = self.scalar_encoder(inputs['scalar'])
        entity_encoded = self.entity_encoder(
            inputs['entity'],
            masks.get('entity') if masks else None
        )
        spatial_encoded = self.spatial_encoder(inputs['spatial'])
        
        # Combine encodings
        return torch.cat([
            scalar_encoded,
            entity_encoded.mean(1),  # Pool entity embeddings
            spatial_encoded.mean([-2, -1])  # Global average pooling
        ], dim=1)
    
    def forward(self,
                inputs: Dict[str, torch.Tensor],
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                masks: Optional[Dict[str, torch.Tensor]] = None,
                temperature: float = 1.0,
                mode: str = 'supervised'
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the AlphaStar model.
        
        Args:
            inputs: Dictionary containing input modalities
            hidden: Optional LSTM hidden state
            masks: Optional masks for different components
            temperature: Temperature for action sampling
            mode: Training mode ('supervised' or 'rl')
            
        Returns:
            Dictionary containing:
            - action_logits: Raw action logits
            - actions: Sampled actions
            - values: State value estimates
            - auxiliary: Optional auxiliary predictions
            - hidden: New LSTM hidden state
        """
        # Encode inputs
        encoded = self.encode_inputs(inputs, masks)
        
        # Process through LSTM core
        core_output, new_hidden = self.core(encoded.unsqueeze(1), hidden)
        core_output = core_output.squeeze(1)
        
        # Generate action predictions
        action_outputs = self.action_heads(
            core_output=core_output,
            entity_states=self.entity_encoder.last_output,
            temperature=temperature,
            mask=masks.get('action') if masks else None
        )
        
        # Compute values
        outputs = {
            **action_outputs,
            'supervised_value': self.supervised_value(core_output),
            'baseline_value': self.baseline_value(core_output),
            'hidden': new_hidden
        }
        
        # Add auxiliary predictions if enabled
        if hasattr(self, 'auxiliary_heads'):
            outputs['auxiliary'] = {
                name: head(core_output)
                for name, head in self.auxiliary_heads.items()
            }
        
        return outputs
    
    def get_action(self,
                   inputs: Dict[str, torch.Tensor],
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   masks: Optional[Dict[str, torch.Tensor]] = None,
                   deterministic: bool = False
                  ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get actions for inference/evaluation.
        
        Args:
            inputs: Dictionary containing input modalities
            hidden: Optional LSTM hidden state
            masks: Optional masks for different components
            deterministic: If True, use argmax instead of sampling
            
        Returns:
            Tuple of (actions_dict, new_hidden_state)
        """
        temperature = 0.0 if deterministic else 1.0
        outputs = self.forward(
            inputs,
            hidden,
            masks,
            temperature=temperature,
            mode='inference'
        )
        
        actions = {}
        new_hidden = outputs.pop('hidden')
        
        # Convert logits to actions
        for key in ['action_type', 'delay', 'queued', 'selected_units',
                   'target_unit', 'target_location']:
            if key in outputs:
                if key in ['delay', 'queued']:
                    actions[key] = torch.sigmoid(outputs[key])
                elif deterministic:
                    actions[key] = outputs[key].argmax(dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=outputs[key])
                    actions[key] = dist.sample()
        
        return actions, new_hidden 