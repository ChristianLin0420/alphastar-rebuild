"""
Tests for the scalar encoder module.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from models.modules.scalar_encoder import ScalarEncoder

def test_scalar_encoder_initialization(test_config):
    """Test scalar encoder initialization."""
    encoder = ScalarEncoder(
        input_dim=test_config['input_dims']['scalar'],
        hidden_dims=test_config['network_params']['scalar_encoder'],
        output_dim=test_config['network_params']['encoder_output']
    )
    
    assert isinstance(encoder, ScalarEncoder)
    # Count number of Linear layers (one for each hidden layer plus output)
    num_linear_layers = sum(1 for m in encoder.network if isinstance(m, torch.nn.Linear))
    assert num_linear_layers == len(test_config['network_params']['scalar_encoder']) + 1

def test_scalar_encoder_forward(test_config, mock_input_batch, device):
    """Test forward pass of scalar encoder."""
    encoder = ScalarEncoder(
        input_dim=test_config['input_dims']['scalar'],
        hidden_dims=test_config['network_params']['scalar_encoder'],
        output_dim=test_config['network_params']['encoder_output']
    ).to(device)
    
    # Forward pass
    scalar_input = mock_input_batch['scalar']
    output = encoder(scalar_input)
    
    # Check output shape
    batch_size = scalar_input.size(0)
    expected_output_dim = test_config['network_params']['encoder_output']
    assert output.size() == (batch_size, expected_output_dim)
    
    # Check output properties
    assert torch.isfinite(output).all()
    assert not torch.isnan(output).any()

def test_scalar_encoder_gradient_flow(test_config, mock_input_batch, device):
    """Test gradient flow through scalar encoder."""
    encoder = ScalarEncoder(
        input_dim=test_config['input_dims']['scalar'],
        hidden_dims=test_config['network_params']['scalar_encoder'],
        output_dim=test_config['network_params']['encoder_output']
    ).to(device)
    
    # Forward and backward pass
    scalar_input = mock_input_batch['scalar']
    output = encoder(scalar_input)
    loss = output.mean()
    loss.backward()
    
    # Check gradients
    for param in encoder.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()

@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_scalar_encoder_batch_sizes(batch_size, test_config, device):
    """Test scalar encoder with different batch sizes."""
    encoder = ScalarEncoder(
        input_dim=test_config['input_dims']['scalar'],
        hidden_dims=test_config['network_params']['scalar_encoder'],
        output_dim=test_config['network_params']['encoder_output']
    ).to(device)
    
    # Create input with different batch size
    scalar_input = torch.randn(batch_size, test_config['input_dims']['scalar']).to(device)
    
    # Forward pass
    output = encoder(scalar_input)
    assert output.size(0) == batch_size
    assert output.size(1) == test_config['network_params']['encoder_output']

def test_scalar_encoder_activation(test_config, mock_input_batch, device):
    """Test activation functions in scalar encoder."""
    encoder = ScalarEncoder(
        input_dim=test_config['input_dims']['scalar'],
        hidden_dims=test_config['network_params']['scalar_encoder'],
        output_dim=test_config['network_params']['encoder_output']
    ).to(device)
    
    # Forward pass
    scalar_input = mock_input_batch['scalar']
    
    # Test intermediate activations
    x = scalar_input
    for i, layer in enumerate(encoder.network):
        x = layer(x)
        if isinstance(layer, torch.nn.ReLU):
            # Check ReLU output is non-negative
            assert (x >= 0).all(), f"ReLU output contains negative values at layer {i}" 