import torch
import pytest
from ..models.scalar_encoder import ScalarEncoder

def test_scalar_encoder_shapes():
    """Test if ScalarEncoder produces correct output shapes."""
    batch_size = 32
    input_dim = 10
    output_dim = 256
    
    encoder = ScalarEncoder(input_dim=input_dim, output_dim=output_dim)
    x = torch.randn(batch_size, input_dim)
    
    output = encoder(x)
    assert output.shape == (batch_size, output_dim)

def test_scalar_encoder_invalid_input():
    """Test if ScalarEncoder properly handles invalid inputs."""
    encoder = ScalarEncoder(input_dim=10)
    
    # Test wrong input dimension
    with pytest.raises(ValueError):
        x = torch.randn(32, 5)  # Wrong input dimension
        encoder(x)
    
    # Test wrong number of dimensions
    with pytest.raises(ValueError):
        x = torch.randn(32, 10, 1)  # 3D tensor instead of 2D
        encoder(x)

def test_scalar_encoder_batch_invariance():
    """Test if ScalarEncoder works with different batch sizes."""
    input_dim = 10
    encoder = ScalarEncoder(input_dim=input_dim)
    
    # Test different batch sizes
    batch_sizes = [1, 16, 32, 64]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, input_dim)
        output = encoder(x)
        assert output.shape == (batch_size, 256)

def test_scalar_encoder_gradient_flow():
    """Test if gradients flow properly through the ScalarEncoder."""
    encoder = ScalarEncoder(input_dim=10)
    x = torch.randn(32, 10, requires_grad=True)
    
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None
    assert all(p.grad is not None for p in encoder.parameters())

def test_scalar_encoder_output_range():
    """Test if ScalarEncoder output values are in a reasonable range."""
    encoder = ScalarEncoder(input_dim=10)
    x = torch.randn(32, 10)
    
    output = encoder(x)
    
    # Check if outputs are not exploding
    assert torch.isfinite(output).all()
    assert output.abs().mean() < 100

@pytest.mark.parametrize("hidden_dims", [
    [128, 128],
    [512, 256, 128],
    [64],
    []
])
def test_scalar_encoder_architectures(hidden_dims):
    """Test if ScalarEncoder works with different architectures."""
    encoder = ScalarEncoder(input_dim=10, hidden_dims=hidden_dims)
    x = torch.randn(32, 10)
    
    output = encoder(x)
    assert output.shape == (32, 256) 