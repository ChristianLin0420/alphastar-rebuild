import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from data.preprocessor import SC2Preprocessor
from configs.training_config import TrainingConfig

class MockObs:
    """Mock SC2 observation for testing."""
    def __init__(self):
        # Create a mock player object with numeric attributes
        class Player:
            def __init__(self):
                self.minerals = 1000
                self.vespene = 500
                self.food_used = 100
                self.food_cap = 200
                self.army_count = 50
                self.worker_count = 30
        
        self.player = Player()
        
        # Create mock units with numeric attributes
        class Unit:
            def __init__(self):
                self.unit_type = 1
                self.alliance = 1
                self.health = 100
                self.shield = 50
                self.energy = 200
                self.x = 10
                self.y = 20
                self.is_selected = 1
        
        self.feature_units = [Unit() for _ in range(10)]
        
        # Create mock minimap with numpy arrays
        class FeatureMinimap:
            def __init__(self):
                self.height_map = np.zeros((64, 64), dtype=np.int32)
                self.visibility_map = np.ones((64, 64), dtype=np.int32)
                self.player_relative = np.ones((64, 64), dtype=np.int32)
        
        self.feature_minimap = FeatureMinimap()

def test_scalar_feature_extraction():
    """Test scalar feature extraction and normalization."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    obs = Mock(observation=MockObs())
    
    scalar_features = preprocessor.extract_scalar_features(obs)
    
    assert isinstance(scalar_features, torch.Tensor)
    assert scalar_features.dim() == 1
    assert torch.all(scalar_features >= 0) and torch.all(scalar_features <= 1)

def test_entity_feature_extraction():
    """Test entity feature extraction and masking."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    obs = Mock(observation=MockObs())
    
    entity_features, mask = preprocessor.extract_entity_features(obs)
    
    assert isinstance(entity_features, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert entity_features.dim() == 2
    assert mask.dim() == 1
    assert torch.sum(mask) == len(obs.observation.feature_units)

def test_spatial_feature_extraction():
    """Test spatial feature extraction and resizing."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    obs = Mock(observation=MockObs())
    
    spatial_features = preprocessor.extract_spatial_features(obs)
    
    assert isinstance(spatial_features, torch.Tensor)
    assert spatial_features.dim() == 3
    assert spatial_features.shape[-2:] == config.SPATIAL_SIZE

def test_full_observation_preprocessing():
    """Test complete observation preprocessing pipeline."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    obs = Mock(observation=MockObs())
    
    processed = preprocessor.preprocess_observation(obs)
    
    assert isinstance(processed, dict)
    assert all(key in processed for key in 
              ['scalar_input', 'entity_input', 'spatial_input', 'entity_mask'])
    assert all(isinstance(v, torch.Tensor) for v in processed.values())

def test_action_preprocessing():
    """Test action preprocessing."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    
    # Mock SC2 action
    action = Mock(
        function=1,  # Some action ID
        queue=False,
        arguments=[(32, 32)],  # Spatial coordinate
        selected_units=[0, 1, 2]  # Selected unit IDs
    )
    
    processed = preprocessor.preprocess_action(action)
    
    assert isinstance(processed, dict)
    assert isinstance(processed['action_type'], torch.Tensor)
    assert isinstance(processed['queued'], torch.Tensor)
    if 'target_location' in processed:
        assert processed['target_location'].shape == config.SPATIAL_SIZE
    if 'selected_units' in processed:
        assert processed['selected_units'].shape == (config.MAX_ENTITIES,)

@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_batch_processing(batch_size):
    """Test preprocessing with different batch sizes."""
    config = TrainingConfig()
    preprocessor = SC2Preprocessor(config)
    
    # Create batch of observations
    batch_obs = [Mock(observation=MockObs()) for _ in range(batch_size)]
    
    # Process each observation
    processed_batch = [
        preprocessor.preprocess_observation(obs)
        for obs in batch_obs
    ]
    
    # Stack tensors
    batch_processed = {
        key: torch.stack([item[key] for item in processed_batch])
        for key in processed_batch[0].keys()
    }
    
    assert all(v.size(0) == batch_size for v in batch_processed.values()) 