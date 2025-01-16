import torch
import torch.nn as nn
from typing import Optional, List

class ResBlock(nn.Module):
    """
    Residual block for the spatial encoder.
    
    Attributes:
        conv1, conv2 (nn.Conv2d): Convolutional layers
        bn1, bn2 (nn.BatchNorm2d): Batch normalization layers
        
    Input Shape:
        - x: (batch_size, channels, height, width)
        
    Output Shape:
        - output: (batch_size, channels, height, width)
    """
    
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width)
        """
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)

class SpatialEncoder(nn.Module):
    """
    Spatial encoder for processing minimap information using ResNet architecture.
    
    Attributes:
        conv1 (nn.Conv2d): Initial convolution layer
        bn1 (nn.BatchNorm2d): Initial batch normalization
        res_blocks (nn.ModuleList): List of residual blocks
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling
        
    Input Shape:
        - x: (batch_size, input_channels, height, width)
        
    Output Shape:
        - encoded: (batch_size, base_channels)
    """
    
    def __init__(self, 
                 input_channels: int,
                 base_channels: int = 64,
                 num_res_blocks: int = 4,
                 dropout: float = 0.1):
        """
        Initialize the SpatialEncoder.
        
        Args:
            input_channels (int): Number of input channels
            base_channels (int): Number of base channels
            num_res_blocks (int): Number of residual blocks
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, base_channels, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(base_channels, dropout)
            for _ in range(num_res_blocks)
        ])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SpatialEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, base_channels)
            
        Raises:
            ValueError: If input tensor shape doesn't match expected dimensions
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
            
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
            
        x = self.global_pool(x)
        return x.flatten(1) 