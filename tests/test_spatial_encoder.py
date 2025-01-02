import torch
import pytest
from ..models.spatial_encoder import SpatialEncoder, ResBlock

def test_spatial_encoder_shapes():
    """Test if SpatialEncoder produces correct output shapes."""
    batch_size = 32
    input_channels = 16
    spatial_size = (128, 128)
    base_channels = 64
    
    encoder = SpatialEncoder(input_channels=input_channels, base_channels=base_channels)
    x = torch.randn(batch_size, input_channels, *spatial_size)
    
    output = encoder(x)
    assert output.shape == (batch_size, base_channels)

def test_spatial_encoder_invalid_input():
    """Test if SpatialEncoder properly handles invalid inputs."""
    encoder = SpatialEncoder(input_channels=16)
    
    # Test wrong input dimension
    with pytest.raises(ValueError):
        x = torch.randn(32, 16, 128)  # Missing spatial dimension
        encoder(x)
    
    # Test wrong number of channels
    with pytest.raises(RuntimeError):
        x = torch.randn(32, 32, 128, 128)  # Wrong number of channels
        encoder(x)

def test_resblock():
    """Test the ResBlock module."""
    channels = 64
    batch_size = 32
    spatial_size = (32, 32)
    
    block = ResBlock(channels)
    x = torch.randn(batch_size, channels, *spatial_size)
    
    output = block(x)
    assert output.shape == x.shape
    
    # Test if residual connection is working
    x_detached = x.clone().detach()
    output_detached = output.clone().detach()
    assert not torch.allclose(x_detached, output_detached)

def test_spatial_encoder_gradient_flow():
    """Test if gradients flow properly through the SpatialEncoder."""
    encoder = SpatialEncoder(input_channels=16)
    x = torch.randn(32, 16, 128, 128, requires_grad=True)
    
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None
    assert all(p.grad is not None for p in encoder.parameters())

@pytest.mark.parametrize("input_size", [
    (64, 64),
    (128, 128),
    (256, 256)
])
def test_spatial_encoder_different_input_sizes(input_size):
    """Test if SpatialEncoder works with different input sizes."""
    encoder = SpatialEncoder(input_channels=16)
    x = torch.randn(32, 16, *input_size)
    
    output = encoder(x)
    assert output.shape == (32, 64)

def test_spatial_encoder_output_range():
    """Test if SpatialEncoder output values are in a reasonable range."""
    encoder = SpatialEncoder(input_channels=16)
    x = torch.randn(32, 16, 128, 128)
    
    output = encoder(x)
    
    # Check if outputs are not exploding
    assert torch.isfinite(output).all()
    assert output.abs().mean() < 100 