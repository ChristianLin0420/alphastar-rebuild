"""
Tests for the AlphaStar model.
"""

import pytest
import torch
import torch.nn.functional as F
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
    # Convert test_config to model expected format
    config = {
        'input_dims': test_config['input_dims'],
        'network_params': test_config['network_params'],
        'action_space': test_config['action_space'],
        'training_params': test_config['training_params']
    }
    
    model = AlphaStar(config)
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
    config = {
        'input_dims': test_config['input_dims'],
        'network_params': test_config['network_params'],
        'action_space': test_config['action_space'],
        'training_params': test_config['training_params']
    }
    
    model = AlphaStar(config).to(device)
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
    max_entities = 100  # From mock_input_batch
    
    assert outputs['action_type'].size() == (batch_size, num_actions)
    assert outputs['delay'].size() == (batch_size, 1)
    assert outputs['queued'].size() == (batch_size, 1)
    assert outputs['selected_units'].size() == (batch_size, max_entities)
    assert outputs['target_unit'].size() == (batch_size, max_entities)
    assert outputs['target_location'].size() == (batch_size, 1, 32, 32)  # Fixed spatial size from SpatialActionHead
    assert outputs['supervised_value'].size() == (batch_size, 1)
    assert outputs['baseline_value'].size() == (batch_size, 1)

def test_alphastar_get_action(test_config, mock_input_batch, device):
    """Test action sampling from AlphaStar model."""
    config = {
        'input_dims': test_config['input_dims'],
        'network_params': test_config['network_params'],
        'action_space': test_config['action_space'],
        'training_params': test_config['training_params']
    }
    
    model = AlphaStar(config).to(device)
    model.eval()
    
    # Test deterministic action selection
    with torch.no_grad():
        outputs = model(
            inputs=mock_input_batch,
            temperature=0.0,
            mode='inference'
        )
        
        # Convert outputs to actions
        actions = {
            'action_type': outputs['action_type'].argmax(dim=-1),
            'delay': torch.sigmoid(outputs['delay']),
            'queued': outputs['queued'],
            'selected_units': outputs['selected_units'],
            'target_unit': outputs['target_unit'],
            'target_location': outputs['target_location']
        }
        
        hidden = outputs.get('hidden')
    
    # Check action structure
    assert 'action_type' in actions
    assert 'delay' in actions
    assert 'queued' in actions
    assert 'selected_units' in actions
    assert 'target_unit' in actions
    assert 'target_location' in actions
    
    # Check hidden state if returned
    if hidden is not None:
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2
        assert all(isinstance(h, torch.Tensor) for h in hidden)
    
    # Test stochastic action selection
    with torch.no_grad():
        outputs = model(
            inputs=mock_input_batch,
            temperature=1.0,
            mode='inference'
        )
        
        # Convert outputs to actions
        actions = {
            'action_type': torch.multinomial(F.softmax(outputs['action_type'], dim=-1), 1).squeeze(-1),
            'delay': torch.sigmoid(outputs['delay']),
            'queued': outputs['queued'],
            'selected_units': outputs['selected_units'],
            'target_unit': outputs['target_unit'],
            'target_location': outputs['target_location']
        }
    
    # Actions should be valid
    assert torch.is_tensor(actions['action_type'])
    assert actions['action_type'].size(0) == mock_input_batch['scalar'].size(0)

def test_alphastar_auxiliary_heads(test_config, mock_input_batch, device):
    """Test auxiliary prediction heads."""
    # Enable auxiliary heads
    config = {
        'input_dims': test_config['input_dims'],
        'network_params': test_config['network_params'],
        'action_space': test_config['action_space'],
        'training_params': {**test_config['training_params'], 'use_auxiliary': True}
    }
    
    model = AlphaStar(config).to(device)
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
    config = {
        'input_dims': test_config['input_dims'],
        'network_params': test_config['network_params'],
        'action_space': test_config['action_space'],
        'training_params': test_config['training_params']
    }
    
    model = AlphaStar(config).to(device)
    
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