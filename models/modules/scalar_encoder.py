import torch
import torch.nn as nn
from typing import List, Optional

class ScalarEncoder(nn.Module):
    """
    Encodes scalar features for the AlphaStar architecture using an MLP.
    
    Attributes:
        network (nn.Sequential): Multi-layer perceptron for encoding scalar features
        
    Input Shape:
        - scalar_input: (batch_size, input_dim)
        
    Output Shape:
        - encoded: (batch_size, output_dim)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int] = [256, 256], 
                 output_dim: int = 256,
                 dropout: float = 0.1):
        """
        Initialize the ScalarEncoder.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Dimension of output features
            dropout (float): Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build MLP layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ScalarEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, output_dim)
            
        Raises:
            ValueError: If input tensor shape doesn't match expected dimensions
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
        if x.size(1) != self.network[0].in_features:
            raise ValueError(
                f"Expected input dimension {self.network[0].in_features}, "
                f"got {x.size(1)}"
            )
        
        return self.network(x) 