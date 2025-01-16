import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module with layer normalization.
    
    Attributes:
        mha (nn.MultiheadAttention): PyTorch's multihead attention layer
        layer_norm (nn.LayerNorm): Layer normalization
        
    Input Shape:
        - x: (seq_len, batch_size, d_model)
        - mask: (batch_size, seq_len) or None
        
    Output Shape:
        - attention_output: (seq_len, batch_size, d_model)
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.mha = nn.MultiheadAttention(d_model, num_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the multi-head attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)
            mask (Optional[torch.Tensor]): Attention mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Attention output of shape (seq_len, batch_size, d_model)
        """
        attn_output, _ = self.mha(x, x, x, key_padding_mask=mask)
        return self.layer_norm(x + attn_output)

class EntityEncoder(nn.Module):
    """
    Transformer-based encoder for processing entity information.
    
    Attributes:
        input_projection (nn.Linear): Projects entity features to transformer dimension
        transformer_layers (nn.ModuleList): List of transformer layers
        output_projection (nn.Linear): Projects final output to desired dimension
        last_output (torch.Tensor): Last computed output, used by action heads
        
    Input Shape:
        - entities: (batch_size, num_entities, input_dim)
        - mask: (batch_size, num_entities) or None
        
    Output Shape:
        - encoded: (batch_size, num_entities, d_model)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 d_model: int = 256, 
                 num_heads: int = 8, 
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the EntityEncoder.
        
        Args:
            input_dim (int): Dimension of input entity features
            d_model (int): Dimension of the transformer
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transformer_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.last_output = None
        
    def forward(self, 
                entities: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the EntityEncoder.
        
        Args:
            entities (torch.Tensor): Entity features of shape (batch_size, num_entities, input_dim)
            mask (Optional[torch.Tensor]): Mask of shape (batch_size, num_entities)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, num_entities, d_model)
            
        Raises:
            ValueError: If input tensor shape doesn't match expected dimensions
        """
        if entities.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {entities.shape}")
            
        # Project entity features
        x = self.input_projection(entities)
        
        # Transformer expects shape: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
            
        # Return to original shape: (batch_size, num_entities, d_model)
        x = x.transpose(0, 1)
        
        # Store output for action heads
        self.last_output = self.output_projection(x)
        
        return self.last_output 