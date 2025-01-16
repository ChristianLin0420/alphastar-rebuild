import pytest
import torch

from models.modules.action_heads import (
    ActionHeads,
    ActionTypeHead,
    PointerNetwork,
    SpatialActionHead
)

def test_action_type_head():
    """Test ActionTypeHead output shapes and values."""
    batch_size = 32
    input_dim = 512
    num_actions = 100
    
    head = ActionTypeHead(input_dim, num_actions)
    x = torch.randn(batch_size, input_dim)
    
    output = head(x)
    assert output.shape == (batch_size, num_actions)
    assert torch.isfinite(output).all()

def test_pointer_network():
    """Test PointerNetwork attention mechanism."""
    batch_size = 32
    query_dim = 512
    key_dim = 256
    num_entities = 64
    
    network = PointerNetwork(query_dim, key_dim)
    query = torch.randn(batch_size, query_dim)
    keys = torch.randn(batch_size, num_entities, key_dim)
    mask = torch.ones(batch_size, num_entities, dtype=torch.bool)
    mask[:, num_entities//2:] = False
    
    # Test without mask
    output = network(query, keys)
    assert output.shape == (batch_size, num_entities)
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size))
    
    # Test with mask
    output_masked = network(query, keys, mask)
    assert output_masked[:, num_entities//2:].sum() == 0

def test_spatial_action_head():
    """Test SpatialActionHead output shapes."""
    batch_size = 32
    input_dim = 512
    spatial_size = (32, 32)
    
    head = SpatialActionHead(input_dim, spatial_size)
    x = torch.randn(batch_size, input_dim)
    
    output = head(x)
    assert output.shape == (batch_size, 1, *spatial_size)
    assert torch.isfinite(output).all()

def test_action_heads_combined():
    """Test complete ActionHeads module."""
    batch_size = 32
    core_dim = 512
    entity_dim = 256
    num_actions = 100
    num_entities = 64
    spatial_size = (32, 32)
    
    heads = ActionHeads(core_dim, entity_dim, num_actions, spatial_size)
    core_output = torch.randn(batch_size, core_dim)
    entity_states = torch.randn(batch_size, num_entities, entity_dim)
    mask = torch.ones(batch_size, num_entities, dtype=torch.bool)
    
    outputs = heads(core_output, entity_states, mask)
    
    # Check all output shapes
    assert outputs['action_type'].shape == (batch_size, num_actions)
    assert outputs['delay'].shape == (batch_size, 1)
    assert outputs['queued'].shape == (batch_size, 1)
    assert outputs['selected_units'].shape == (batch_size, num_entities)
    assert outputs['target_unit'].shape == (batch_size, num_entities)
    assert outputs['target_location'].shape == (batch_size, 1, *spatial_size)
    assert outputs['value'].shape == (batch_size, 1)
    
    # Check value ranges
    assert torch.all(outputs['queued'] >= 0) and torch.all(outputs['queued'] <= 1)
    assert torch.all(outputs['value'] >= -1) and torch.all(outputs['value'] <= 1)
    assert torch.allclose(outputs['selected_units'].sum(dim=1), torch.ones(batch_size))
    assert torch.allclose(outputs['target_unit'].sum(dim=1), torch.ones(batch_size))

def test_action_heads_gradient_flow():
    """Test if gradients flow properly through all heads."""
    heads = ActionHeads(512, 256, 100, (32, 32))
    core_output = torch.randn(32, 512, requires_grad=True)
    entity_states = torch.randn(32, 64, 256, requires_grad=True)
    
    outputs = heads(core_output, entity_states)
    loss = sum(output.sum() for output in outputs.values())
    loss.backward()
    
    assert core_output.grad is not None
    assert entity_states.grad is not None
    assert all(p.grad is not None for p in heads.parameters())

@pytest.mark.parametrize("batch_size", [1, 16, 64])
def test_action_heads_batch_sizes(batch_size):
    """Test if ActionHeads works with different batch sizes."""
    heads = ActionHeads(512, 256, 100, (32, 32))
    core_output = torch.randn(batch_size, 512)
    entity_states = torch.randn(batch_size, 64, 256)
    
    outputs = heads(core_output, entity_states)
    assert all(v.size(0) == batch_size for v in outputs.values()) 