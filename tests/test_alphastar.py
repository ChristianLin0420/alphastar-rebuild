import torch
import pytest
from ..models.alphastar import AlphaStar
from ..configs.training_config import TrainingConfig

@pytest.fixture
def model_input():
    """Create dummy model input."""
    batch_size = 32
    config = TrainingConfig()
    
    return {
        'scalar_input': torch.randn(batch_size, config.SCALAR_INPUT_DIM),
        'entity_input': torch.randn(batch_size, config.MAX_ENTITIES, config.ENTITY_INPUT_DIM),
        'spatial_input': torch.randn(batch_size, config.SPATIAL_INPUT_CHANNELS, *config.SPATIAL_SIZE),
        'entity_mask': torch.ones(batch_size, config.MAX_ENTITIES, dtype=torch.bool)
    }

def test_alphastar_output_shapes():
    """Test if AlphaStar produces correct output shapes."""
    config = TrainingConfig()
    model = AlphaStar(config)
    inputs = model_input()
    
    outputs = model(inputs)
    
    # Check output shapes
    assert outputs['action_type'].shape == (32, config.NUM_ACTIONS)
    assert outputs['delay'].shape == (32, 1)
    assert outputs['queued'].shape == (32, 1)
    assert outputs['selected_units'].shape == (32, config.MAX_ENTITIES)
    assert outputs['target_unit'].shape == (32, config.MAX_ENTITIES)
    assert outputs['target_location'].shape == (32, 1, *config.SPATIAL_SIZE)
    assert outputs['value'].shape == (32, 1)
    assert isinstance(outputs['hidden'], tuple)
    assert len(outputs['hidden']) == 2

def test_alphastar_get_action():
    """Test action extraction for inference."""
    config = TrainingConfig()
    model = AlphaStar(config)
    inputs = model_input()
    
    actions, hidden = model.get_action(inputs, deterministic=True)
    
    # Check action shapes
    assert isinstance(actions['action_type'], torch.Tensor)
    assert actions['action_type'].shape == (32,)
    assert actions['delay'].shape == (32, 1)
    assert actions['queued'].shape == (32, 1)
    assert actions['selected_units'].shape == (32,)
    assert actions['target_unit'].shape == (32,)
    assert actions['target_location'].shape == (32, 2)
    
    # Check hidden state
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2

def test_alphastar_invalid_input():
    """Test if AlphaStar properly handles invalid inputs."""
    config = TrainingConfig()
    model = AlphaStar(config)
    
    # Test missing input
    with pytest.raises(AssertionError):
        model({'scalar_input': torch.randn(32, config.SCALAR_INPUT_DIM)})
    
    # Test wrong dimensions
    invalid_input = model_input()
    invalid_input['scalar_input'] = invalid_input['scalar_input'].unsqueeze(0)
    with pytest.raises(AssertionError):
        model(invalid_input)

def test_alphastar_gradient_flow():
    """Test if gradients flow properly through the model."""
    config = TrainingConfig()
    model = AlphaStar(config)
    inputs = model_input()
    
    outputs = model(inputs)
    loss = sum(output.mean() for output in outputs.values() if isinstance(output, torch.Tensor))
    loss.backward()
    
    assert all(p.grad is not None for p in model.parameters())

@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_alphastar_batch_sizes(batch_size):
    """Test if AlphaStar works with different batch sizes."""
    config = TrainingConfig()
    model = AlphaStar(config)
    
    inputs = {
        'scalar_input': torch.randn(batch_size, config.SCALAR_INPUT_DIM),
        'entity_input': torch.randn(batch_size, config.MAX_ENTITIES, config.ENTITY_INPUT_DIM),
        'spatial_input': torch.randn(batch_size, config.SPATIAL_INPUT_CHANNELS, *config.SPATIAL_SIZE),
        'entity_mask': torch.ones(batch_size, config.MAX_ENTITIES, dtype=torch.bool)
    }
    
    outputs = model(inputs)
    assert all(v.size(0) == batch_size for v in outputs.values() if isinstance(v, torch.Tensor)) 