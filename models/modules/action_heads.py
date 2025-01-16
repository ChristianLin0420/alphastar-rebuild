import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ActionTypeHead(nn.Module):
    """
    Head for predicting action type probabilities.
    
    Input Shape:
        - x: (batch_size, input_dim)
        
    Output Shape:
        - action_logits: (batch_size, num_actions)
    """
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PointerNetwork(nn.Module):
    """
    Pointer network for selecting entities.
    
    Input Shape:
        - query: (batch_size, query_dim)
        - keys: (batch_size, num_entities, key_dim)
        - mask: (batch_size, num_entities) or None
        
    Output Shape:
        - attention_weights: (batch_size, num_entities)
    """
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))
        
    def forward(self, 
                query: torch.Tensor, 
                keys: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Project query and keys
        query = self.query_proj(query).unsqueeze(1)  # [B, 1, H]
        keys = self.key_proj(keys)  # [B, N, H]
        
        # Calculate attention scores
        scores = torch.tanh(query + keys) @ self.v  # [B, N]
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
            
        return F.softmax(scores, dim=-1)

class SpatialActionHead(nn.Module):
    """
    Head for predicting spatial actions using deconvolution.
    
    Input Shape:
        - x: (batch_size, input_dim)
        
    Output Shape:
        - spatial_logits: (batch_size, 1, height, width)
    """
    def __init__(self, input_dim: int, spatial_size: Tuple[int, int]):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ActionHeads(nn.Module):
    """
    Combined action heads for AlphaStar.
    
    Attributes:
        action_type_head: Predicts action type
        delay_head: Predicts action delay
        queued_head: Predicts if action is queued
        selected_units_head: Selects units using pointer network
        target_unit_head: Selects target unit using pointer network
        target_location_head: Predicts spatial target location
        value_head: Predicts state value
        
    Input Shape:
        - core_output: (batch_size, core_dim)
        - entity_states: (batch_size, num_entities, entity_dim)
        - mask: (batch_size, num_entities) or None
        
    Output Shape:
        Dictionary containing:
        - action_type: (batch_size, num_actions)
        - delay: (batch_size, 1)
        - queued: (batch_size, 1)
        - selected_units: (batch_size, num_entities)
        - target_unit: (batch_size, num_entities)
        - target_location: (batch_size, 1, height, width)
        - value: (batch_size, 1)
    """
    
    def __init__(self, 
                 core_dim: int,
                 entity_dim: int,
                 num_actions: int,
                 spatial_size: Tuple[int, int]):
        super().__init__()
        
        self.action_type_head = ActionTypeHead(core_dim, num_actions)
        
        self.delay_head = nn.Sequential(
            nn.Linear(core_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.queued_head = nn.Sequential(
            nn.Linear(core_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.selected_units_head = PointerNetwork(core_dim, entity_dim)
        self.target_unit_head = PointerNetwork(core_dim, entity_dim)
        self.target_location_head = SpatialActionHead(core_dim, spatial_size)
        
        self.value_head = nn.Sequential(
            nn.Linear(core_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, 
                core_output: torch.Tensor,
                entity_states: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all action heads.
        
        Args:
            core_output: Core LSTM output
            entity_states: Entity feature states
            mask: Entity mask for pointer networks
            
        Returns:
            Dictionary containing all action outputs
        """
        return {
            'action_type': self.action_type_head(core_output),
            'delay': self.delay_head(core_output),
            'queued': self.queued_head(core_output),
            'selected_units': self.selected_units_head(core_output, entity_states, mask),
            'target_unit': self.target_unit_head(core_output, entity_states, mask),
            'target_location': self.target_location_head(core_output),
            'value': self.value_head(core_output)
        } 