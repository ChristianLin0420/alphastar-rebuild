import torch
import torch.nn as nn
from typing import Tuple, Optional

class AlphaStarCore(nn.Module):
    """
    Core LSTM module for the AlphaStar architecture.
    
    Attributes:
        lstm (nn.LSTM): Multi-layer LSTM
        layer_norm (nn.LayerNorm): Layer normalization
        
    Input Shape:
        - x: (batch_size, seq_len, input_dim)
        - hidden: (Optional) Tuple of (h_0, c_0) each of shape (num_layers, batch_size, hidden_dim)
        
    Output Shape:
        - output: (batch_size, seq_len, hidden_dim)
        - hidden: Tuple of (h_n, c_n) each of shape (num_layers, batch_size, hidden_dim)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 512, 
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize the AlphaStarCore.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of LSTM hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Store dimensions for initialization
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, 
                x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the core LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Initial hidden state
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: 
                - Output tensor of shape (batch_size, seq_len, hidden_dim)
                - Tuple of final hidden states
                
        Raises:
            ValueError: If input tensor shape doesn't match expected dimensions
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
            
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
            
        # Process through LSTM
        lstm_out, hidden_state = self.lstm(x, hidden)
        
        # Apply layer normalization
        normalized_out = self.layer_norm(lstm_out)
        
        return normalized_out, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial hidden state (h_0, c_0)
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h_0, c_0) 