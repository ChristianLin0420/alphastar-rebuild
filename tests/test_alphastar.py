"""
Tests for the AlphaStar model.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from models.alphastar import AlphaStar
from models.modules.core import AlphaStarCore
from models.modules.scalar_encoder import ScalarEncoder
from models.modules.entity_encoder import EntityEncoder
from models.modules.spatial_encoder import SpatialEncoder
from models.modules.action_heads import ActionHeads

def test_alphastar_initialization(test_config):
    """Test AlphaStar model initialization."""
    model = AlphaStar(test_config)
    assert isinstance(model, AlphaStar)
    
    # Check all components are initialized
    assert hasattr(model, 'scalar_encoder')
    assert hasattr(model, 'entity_encoder')
    assert hasattr(model, 'spatial_encoder')
    assert hasattr(model, 'core')
    assert hasattr(model, 'action_heads')
    assert hasattr(model, 'supervised_value')
    assert hasattr(model, 'baseline_value')

def test_alphastar_forward(test_config, mock_input_batch, device):
    """Test forward pass of AlphaStar model."""
    model = AlphaStar(test_config).to(device)
    model.train()
    
    # Forward pass
    outputs = model(
        inputs=mock_input_batch,
        mode='supervised'
    )
    
    # Check output structure
    assert 'action_type' in outputs
    assert 'delay' in outputs
    assert 'queued' in outputs
    assert 'selected_units' in outputs
    assert 'target_unit' in outputs
    assert 'target_location' in outputs
    assert 'supervised_value' in outputs
    assert 'baseline_value' in outputs
    
    # Check output shapes
    batch_size = mock_input_batch['scalar'].size(0)
    num_actions = test_config['action_space']['num_actions']
    
    assert outputs['action_type'].size() == (batch_size, num_actions)
    assert outputs['delay'].size() == (batch_size, 1)
    assert outputs['queued'].size() == (batch_size, 1)
    assert outputs['selected_units'].size() == (batch_size, 100)  # max_entities
    assert outputs['target_unit'].size() == (batch_size, 100)  # max_entities
    assert outputs['target_location'].size() == (batch_size, 2, 64, 64)  # spatial size
    assert outputs['supervised_value'].size() == (batch_size, 1)
    assert outputs['baseline_value'].size() == (batch_size, 1)

def test_alphastar_get_action(test_config, mock_input_batch, device):
    """Test action sampling from AlphaStar model."""
    model = AlphaStar(test_config).to(device)
    model.eval()
    
    # Test deterministic action selection
    actions, hidden = model.get_action(
        inputs=mock_input_batch,
        deterministic=True
    )
    
    # Check action structure
    assert 'action_type' in actions
    assert 'delay' in actions
    assert 'queued' in actions
    assert 'selected_units' in actions
    assert 'target_unit' in actions
    assert 'target_location' in actions
    
    # Check hidden state
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2
    assert all(isinstance(h, torch.Tensor) for h in hidden)
    
    # Test stochastic action selection
    actions, hidden = model.get_action(
        inputs=mock_input_batch,
        deterministic=False
    )
    
    # Actions should be different when sampled stochastically
    assert torch.is_tensor(actions['action_type'])
    assert actions['action_type'].size(0) == mock_input_batch['scalar'].size(0)

def test_alphastar_auxiliary_heads(test_config, mock_input_batch, device):
    """Test auxiliary prediction heads."""
    # Enable auxiliary heads
    test_config['training_params']['use_auxiliary'] = True
    model = AlphaStar(test_config).to(device)
    model.train()
    
    outputs = model(
        inputs=mock_input_batch,
        mode='supervised'
    )
    
    # Check auxiliary outputs
    assert 'auxiliary' in outputs
    assert 'build_order' in outputs['auxiliary']
    assert 'unit_counts' in outputs['auxiliary']
    
    batch_size = mock_input_batch['scalar'].size(0)
    assert outputs['auxiliary']['build_order'].size() == (batch_size, test_config['action_space']['build_order_size'])
    assert outputs['auxiliary']['unit_counts'].size() == (batch_size, test_config['action_space']['unit_type_size'])

def test_alphastar_training_mode(test_config, mock_input_batch, mock_target_batch, device):
    """Test model in different training modes."""
    model = AlphaStar(test_config).to(device)
    
    # Test supervised mode
    model.train()
    outputs = model(inputs=mock_input_batch, mode='supervised')
    assert 'supervised_value' in outputs
    
    # Test RL mode
    outputs = model(inputs=mock_input_batch, mode='rl')
    assert 'baseline_value' in outputs
    
    # Test with temperature scaling
    outputs = model(inputs=mock_input_batch, temperature=0.5)
    assert all(torch.isfinite(v).all() for v in outputs.values() if torch.is_tensor(v)) 