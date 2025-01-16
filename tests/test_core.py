import pytest
import torch

from models.modules.core import AlphaStarCore
from models.modules.scalar_encoder import ScalarEncoder
from models.modules.entity_encoder import EntityEncoder
from models.modules.spatial_encoder import SpatialEncoder

def test_core_shapes():
    """Test if AlphaStarCore produces correct output shapes."""
    batch_size = 32
    seq_len = 10
    input_dim = 256
    hidden_dim = 512
    num_layers = 3
    
    core = AlphaStarCore(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    x = torch.randn(batch_size, seq_len, input_dim)
    
    output, (h_n, c_n) = core(x)
    
    # Check output shapes
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert h_n.shape == (num_layers, batch_size, hidden_dim)
    assert c_n.shape == (num_layers, batch_size, hidden_dim)

def test_core_hidden_state_initialization():
    """Test if hidden state initialization works correctly."""
    core = AlphaStarCore(input_dim=256)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    h_0, c_0 = core.init_hidden(batch_size, device)
    
    assert h_0.shape == (core.num_layers, batch_size, core.hidden_dim)
    assert c_0.shape == (core.num_layers, batch_size, core.hidden_dim)
    assert h_0.device == device
    assert c_0.device == device

def test_core_with_provided_hidden():
    """Test if AlphaStarCore works with provided hidden state."""
    core = AlphaStarCore(input_dim=256)
    batch_size = 32
    seq_len = 10
    
    x = torch.randn(batch_size, seq_len, 256)
    hidden = core.init_hidden(batch_size, x.device)
    
    output, new_hidden = core(x, hidden)
    
    # Check if hidden state is updated
    assert not torch.allclose(hidden[0], new_hidden[0])
    assert not torch.allclose(hidden[1], new_hidden[1])

def test_core_invalid_input():
    """Test if AlphaStarCore properly handles invalid inputs."""
    core = AlphaStarCore(input_dim=256)
    
    # Test wrong input dimension
    with pytest.raises(ValueError):
        x = torch.randn(32, 256)  # Missing sequence dimension
        core(x)
    
    # Test wrong input size
    with pytest.raises(RuntimeError):
        x = torch.randn(32, 10, 128)  # Wrong input dimension
        core(x)

def test_core_gradient_flow():
    """Test if gradients flow properly through the AlphaStarCore."""
    core = AlphaStarCore(input_dim=256)
    x = torch.randn(32, 10, 256, requires_grad=True)
    
    output, _ = core(x)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    assert x.grad is not None
    assert all(p.grad is not None for p in core.parameters())

@pytest.mark.parametrize("seq_len", [1, 5, 20])
def test_core_different_sequence_lengths(seq_len):
    """Test if AlphaStarCore works with different sequence lengths."""
    core = AlphaStarCore(input_dim=256)
    x = torch.randn(32, seq_len, 256)
    
    output, _ = core(x)
    assert output.shape == (32, seq_len, 512)

def test_core_state_consistency():
    """Test if hidden state is consistently updated."""
    core = AlphaStarCore(input_dim=256)
    batch_size = 32
    
    # Set model to eval mode to ensure deterministic behavior
    core.eval()
    
    # Initial sequence
    x1 = torch.randn(batch_size, 5, 256)
    output1, hidden1 = core(x1)
    
    # Continue with new sequence
    x2 = torch.randn(batch_size, 5, 256)
    output2, hidden2 = core(x2, hidden1)
    
    # Process both sequences at once
    x_combined = torch.cat([x1, x2], dim=1)
    output_combined, hidden_combined = core(x_combined)
    
    # Check if outputs match
    # Compare only the last timestep outputs since LSTM states affect the entire sequence
    assert torch.allclose(
        output_combined[:, -1, :],
        output2[:, -1, :],
        rtol=1e-4,
        atol=1e-4
    ), "Last timestep outputs should match"
    
    # Check if hidden states match
    assert all(
        torch.allclose(h1, h2, rtol=1e-4, atol=1e-4)
        for h1, h2 in zip(hidden2, hidden_combined)
    ), "Hidden states should match" 