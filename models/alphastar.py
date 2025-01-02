import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .scalar_encoder import ScalarEncoder
from .entity_encoder import EntityEncoder
from .spatial_encoder import SpatialEncoder
from .core import AlphaStarCore
from .action_heads import ActionHeads

class AlphaStar(nn.Module):
    """
    Complete AlphaStar architecture combining all components.
    
    Attributes:
        scalar_encoder: Encodes scalar features
        entity_encoder: Encodes entity features
        spatial_encoder: Encodes spatial features
        core: LSTM core for temporal processing
        action_heads: Various action prediction heads
        
    Input Format:
        Dictionary containing:
        - scalar_input: (batch_size, scalar_input_dim)
        - entity_input: (batch_size, max_entities, entity_input_dim)
        - spatial_input: (batch_size, channels, height, width)
        - entity_mask: (batch_size, max_entities)
        
    Output Format:
        Dictionary containing:
        - action_type: (batch_size, num_actions)
        - delay: (batch_size, 1)
        - queued: (batch_size, 1)
        - selected_units: (batch_size, max_entities)
        - target_unit: (batch_size, max_entities)
        - target_location: (batch_size, 1, height, width)
        - value: (batch_size, 1)
    """
    
    def __init__(self, config):
        """
        Initialize AlphaStar model.
        
        Args:
            config: Training configuration object containing model parameters
        """
        super().__init__()
        
        # Feature encoders
        self.scalar_encoder = ScalarEncoder(
            input_dim=config.SCALAR_INPUT_DIM,
            hidden_dims=[256, 256],
            output_dim=256
        )
        
        self.entity_encoder = EntityEncoder(
            input_dim=config.ENTITY_INPUT_DIM,
            d_model=256,
            num_heads=config.TRANSFORMER_NUM_HEADS,
            num_layers=config.TRANSFORMER_NUM_LAYERS
        )
        
        self.spatial_encoder = SpatialEncoder(
            input_channels=config.SPATIAL_INPUT_CHANNELS,
            base_channels=64,
            num_res_blocks=4
        )
        
        # Calculate total encoded dimension
        encoded_dim = 256 + 256 + 64  # Scalar + Entity + Spatial
        
        # Core LSTM
        self.core = AlphaStarCore(
            input_dim=encoded_dim,
            hidden_dim=config.LSTM_HIDDEN_DIM,
            num_layers=config.LSTM_NUM_LAYERS
        )
        
        # Action heads
        self.action_heads = ActionHeads(
            core_dim=config.LSTM_HIDDEN_DIM,
            entity_dim=config.ENTITY_INPUT_DIM,
            num_actions=config.NUM_ACTIONS,
            spatial_size=config.SPATIAL_SIZE
        )
        
        # Save config for validation
        self.config = config
        
    def _validate_input(self, model_input: Dict[str, torch.Tensor]):
        """Validate input tensors."""
        assert 'scalar_input' in model_input, "Missing scalar_input"
        assert 'entity_input' in model_input, "Missing entity_input"
        assert 'spatial_input' in model_input, "Missing spatial_input"
        
        scalar_input = model_input['scalar_input']
        entity_input = model_input['entity_input']
        spatial_input = model_input['spatial_input']
        
        assert scalar_input.dim() == 2, f"Expected 2D scalar_input, got shape {scalar_input.shape}"
        assert entity_input.dim() == 3, f"Expected 3D entity_input, got shape {entity_input.shape}"
        assert spatial_input.dim() == 4, f"Expected 4D spatial_input, got shape {spatial_input.shape}"
        
        assert scalar_input.size(1) == self.config.SCALAR_INPUT_DIM
        assert entity_input.size(2) == self.config.ENTITY_INPUT_DIM
        assert spatial_input.size(1) == self.config.SPATIAL_INPUT_CHANNELS
        assert spatial_input.shape[2:] == self.config.SPATIAL_SIZE
        
    def forward(self, 
                model_input: Dict[str, torch.Tensor],
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the AlphaStar model.
        
        Args:
            model_input: Dictionary containing input tensors
            hidden: Optional LSTM hidden state
            
        Returns:
            Dictionary containing action predictions and value
        """
        # Validate input
        self._validate_input(model_input)
        
        # Extract inputs
        scalar_input = model_input['scalar_input']
        entity_input = model_input['entity_input']
        spatial_input = model_input['spatial_input']
        entity_mask = model_input.get('entity_mask')
        
        # Encode inputs
        scalar_encoded = self.scalar_encoder(scalar_input)
        entity_encoded = self.entity_encoder(entity_input, entity_mask)
        spatial_encoded = self.spatial_encoder(spatial_input)
        
        # Concatenate encoded features
        encoded = torch.cat([
            scalar_encoded,
            entity_encoded.mean(1),  # Average over entities
            spatial_encoded
        ], dim=1)
        
        # Process through core LSTM
        core_output, new_hidden = self.core(encoded.unsqueeze(1), hidden)
        core_output = core_output.squeeze(1)
        
        # Generate action predictions
        action_outputs = self.action_heads(
            core_output=core_output,
            entity_states=entity_encoded,
            mask=entity_mask
        )
        
        # Add hidden state to outputs
        action_outputs['hidden'] = new_hidden
        
        return action_outputs
    
    def get_action(self, 
                   model_input: Dict[str, torch.Tensor],
                   hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                   deterministic: bool = False
                  ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action for inference/evaluation.
        
        Args:
            model_input: Dictionary containing input tensors
            hidden: Optional LSTM hidden state
            deterministic: If True, use argmax instead of sampling
            
        Returns:
            Tuple of (action_dict, new_hidden)
        """
        outputs = self.forward(model_input, hidden)
        new_hidden = outputs.pop('hidden')
        
        # Convert logits to actions
        actions = {}
        
        # Action type
        if deterministic:
            actions['action_type'] = outputs['action_type'].argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=outputs['action_type'])
            actions['action_type'] = dist.sample()
        
        # Delay and queued
        actions['delay'] = outputs['delay']
        actions['queued'] = (outputs['queued'] > 0.5).float()
        
        # Selected units and target unit
        actions['selected_units'] = outputs['selected_units'].argmax(dim=-1)
        actions['target_unit'] = outputs['target_unit'].argmax(dim=-1)
        
        # Target location
        target_loc = outputs['target_location'].view(outputs['target_location'].size(0), -1)
        if deterministic:
            target_idx = target_loc.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=target_loc)
            target_idx = dist.sample()
        
        h, w = self.config.SPATIAL_SIZE
        actions['target_location'] = torch.stack([
            target_idx % w,
            target_idx // w
        ], dim=-1)
        
        return actions, new_hidden 