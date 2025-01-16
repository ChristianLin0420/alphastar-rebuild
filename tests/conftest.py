"""
Shared test fixtures and configurations.
"""

import os
import pytest
import torch
import yaml
from pathlib import Path

@pytest.fixture
def device():
    """Get the default device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def batch_size():
    """Default batch size for testing."""
    return 8

@pytest.fixture
def test_config():
    """Load test configuration."""
    config = {
        'input_dims': {
            'scalar': 32,
            'entity': 64,
            'spatial': 16
        },
        'network_params': {
            'scalar_encoder': [64, 32],
            'encoder_output': 32,
            'transformer_dim': 64,
            'transformer_heads': 2,
            'transformer_layers': 2,
            'dropout': 0.1,
            'spatial_channels': 16,
            'num_res_blocks': 2,
            'lstm_dim': 128,
            'lstm_layers': 2
        },
        'action_space': {
            'num_actions': 100,
            'spatial_size': [64, 64],
            'build_order_size': 20,
            'unit_type_size': 32
        },
        'training_params': {
            'learning_rate': 1e-4,
            'adam_betas': [0.9, 0.999],
            'max_grad_norm': 10.0,
            'use_auxiliary': True,
            'use_league': True,
            'log_dir': 'tests/test_logs',
            'sl_batch_size': 32,
            'sl_num_epochs': 2,
            'sl_warmup_steps': 100,
            'num_workers': 2,
            'pin_memory': True
        }
    }
    return config

@pytest.fixture
def mock_input_batch(batch_size, test_config, device):
    """Create mock input batch for testing."""
    scalar_dim = test_config['input_dims']['scalar']
    entity_dim = test_config['input_dims']['entity']
    spatial_dim = test_config['input_dims']['spatial']
    max_entities = 100  # Maximum number of entities for testing
    
    return {
        'scalar': torch.randn(batch_size, scalar_dim).to(device),
        'entity': torch.randn(batch_size, max_entities, entity_dim).to(device),
        'spatial': torch.randn(batch_size, spatial_dim, 64, 64).to(device),
        'entity_mask': torch.ones(batch_size, max_entities).to(device),
        'valid_actions': torch.ones(batch_size, test_config['action_space']['num_actions']).to(device)
    }

@pytest.fixture
def mock_target_batch(batch_size, test_config, device):
    """Create mock target batch for testing."""
    return {
        'action_type': torch.randint(0, test_config['action_space']['num_actions'], (batch_size,)).to(device),
        'delay': torch.rand(batch_size, 1).to(device),
        'queued': torch.randint(0, 2, (batch_size, 1)).float().to(device),
        'selected_units': torch.randint(0, 100, (batch_size,)).to(device),
        'target_unit': torch.randint(0, 100, (batch_size,)).to(device),
        'target_location': torch.randint(0, 64, (batch_size, 2)).to(device),
        'value': torch.rand(batch_size, 1).to(device)
    } 