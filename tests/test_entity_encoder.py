import torch
import pytest
from ..models.entity_encoder import EntityEncoder, MultiHeadAttention

def test_entity_encoder_shapes():
    """Test if EntityEncoder produces correct output shapes."""
    batch_size = 32
    num_entities = 64
    input_dim = 16
    d_model = 256
    
    encoder = EntityEncoder(input_dim=input_dim, d_model=d_model)
    x = torch.randn(batch_size, num_entities, input_dim)
    mask = torch.ones(batch_size, num_entities, dtype=torch.bool)
    
    output = encoder(x, mask)
    assert output.shape == (batch_size, num_entities, d_model)

def test_entity_encoder_masking():
    """Test if EntityEncoder properly handles attention masking."""
    encoder = EntityEncoder(input_dim=16)
    x = torch.randn(32, 64, 16)
    
    # Create mask where half the entities are masked
    mask = torch.ones(32, 64, dtype=torch.bool)
    mask[:, 32:] = False
    
    output_masked = encoder(x, mask)
    output_unmasked = encoder(x)
    
    # Check if masked and unmasked outputs are different
    assert not torch.allclose(output_masked, output_unmasked)

def test_entity_encoder_invalid_input():
    """Test if EntityEncoder properly handles invalid inputs."""
    encoder = EntityEncoder(input_dim=16)
    
    # Test wrong input dimension
    with pytest.raises(ValueError):
        x = torch.randn(32, 16)  # Missing entity dimension
        encoder(x)
    
    # Test wrong feature dimension
    with pytest.raises(RuntimeError):
        x = torch.randn(32, 64, 32)  # Wrong feature dimension
        encoder(x)

def test_entity_encoder_gradient_flow():
    """Test if gradients flow properly through the EntityEncoder."""
    encoder = EntityEncoder(input_dim=16)
    x = torch.randn(32, 64, 16, requires_grad=True)
    
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None
    assert all(p.grad is not None for p in encoder.parameters())

def test_multi_head_attention():
    """Test the MultiHeadAttention module."""
    d_model = 256
    num_heads = 8
    seq_len = 64
    batch_size = 32
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(seq_len, batch_size, d_model)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    output = mha(x, mask)
    assert output.shape == (seq_len, batch_size, d_model)

@pytest.mark.parametrize("num_layers,num_heads", [
    (1, 4),
    (2, 8),
    (4, 16),
])
def test_entity_encoder_architectures(num_layers, num_heads):
    """Test if EntityEncoder works with different architectures."""
    encoder = EntityEncoder(
        input_dim=16,
        num_layers=num_layers,
        num_heads=num_heads
    )
    x = torch.randn(32, 64, 16)
    
    output = encoder(x)
    assert output.shape == (32, 64, 256) 