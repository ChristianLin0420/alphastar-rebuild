import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pysc2.lib import features, actions

class SC2Preprocessor:
    """
    Preprocessor for StarCraft II observations and actions.
    
    This class handles:
    1. Feature extraction from SC2 observations
    2. Normalization of numerical values
    3. Conversion between SC2 actions and model actions
    """
    
    def __init__(self, config):
        self.config = config
        
        # Define feature specs
        self.scalar_features = [
            'minerals',
            'vespene',
            'food_used',
            'food_cap',
            'army_count',
            'worker_count',
            'idle_worker_count',
            'army_value',
            'structure_value'
        ]
        
        self.entity_features = [
            'unit_type',
            'alliance',
            'health',
            'shield',
            'energy',
            'x',
            'y',
            'is_selected',
            'on_screen',
            'cargo_space_taken'
        ]
        
        # Setup normalization parameters
        self.setup_normalizers()
        
    def setup_normalizers(self):
        """Initialize normalization parameters for features."""
        self.scalar_normalizers = {
            'minerals': 10000.0,
            'vespene': 10000.0,
            'food_used': 200.0,
            'food_cap': 200.0,
            'army_count': 200.0,
            'worker_count': 100.0,
            'idle_worker_count': 100.0,
            'army_value': 10000.0,
            'structure_value': 10000.0
        }
        
        self.entity_normalizers = {
            'health': 1000.0,
            'shield': 1000.0,
            'energy': 200.0,
            'x': float(self.config.SPATIAL_SIZE[0]),
            'y': float(self.config.SPATIAL_SIZE[1])
        }
    
    def preprocess_observation(self, obs) -> Dict[str, torch.Tensor]:
        """
        Convert SC2 observation to model input format.
        
        Args:
            obs: PySC2 observation object
            
        Returns:
            Dict containing preprocessed features:
                - scalar_input: (batch_size, num_scalar_features)
                - entity_input: (batch_size, max_entities, entity_feature_dim)
                - spatial_input: (batch_size, channels, height, width)
                - entity_mask: (batch_size, max_entities)
        """
        # Process scalar features
        scalar_features = self.extract_scalar_features(obs)
        
        # Process entity features
        entity_features, entity_mask = self.extract_entity_features(obs)
        
        # Process spatial features
        spatial_features = self.extract_spatial_features(obs)
        
        return {
            'scalar_input': scalar_features,
            'entity_input': entity_features,
            'spatial_input': spatial_features,
            'entity_mask': entity_mask
        }
    
    def extract_scalar_features(self, obs) -> torch.Tensor:
        """Extract and normalize scalar features."""
        features = []
        player = obs.observation.player
        
        for feature_name in self.scalar_features:
            value = getattr(player, feature_name, 0)
            normalized_value = value / self.scalar_normalizers[feature_name]
            features.append(normalized_value)
            
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_entity_features(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and normalize entity features."""
        max_entities = self.config.MAX_ENTITIES
        feature_dim = len(self.entity_features)
        
        features = torch.zeros(max_entities, feature_dim)
        mask = torch.zeros(max_entities, dtype=torch.bool)
        
        for i, unit in enumerate(obs.observation.feature_units[:max_entities]):
            feature_vector = []
            for feature_name in self.entity_features:
                value = getattr(unit, feature_name, 0)
                if feature_name in self.entity_normalizers:
                    value = value / self.entity_normalizers[feature_name]
                feature_vector.append(value)
            
            features[i] = torch.tensor(feature_vector)
            mask[i] = True
            
        return features, mask
    
    def extract_spatial_features(self, obs) -> torch.Tensor:
        """Extract and normalize spatial features."""
        minimap = obs.observation.feature_minimap
        
        # Convert feature layers to tensor
        layers = []
        for feature in features.MINIMAP_FEATURES:
            layer = getattr(minimap, feature.name, None)
            if layer is not None:
                if feature.type == features.FeatureType.CATEGORICAL:
                    # One-hot encode categorical features
                    layer = np.eye(feature.scale)[layer]
                else:
                    # Normalize numerical features
                    layer = layer.astype(np.float32) / feature.scale
                layers.append(layer)
        
        spatial_tensor = torch.tensor(np.stack(layers), dtype=torch.float32)
        
        # Ensure correct spatial dimensions
        if spatial_tensor.shape[-2:] != self.config.SPATIAL_SIZE:
            spatial_tensor = torch.nn.functional.interpolate(
                spatial_tensor.unsqueeze(0),
                size=self.config.SPATIAL_SIZE,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
        return spatial_tensor
    
    def preprocess_action(self, action) -> Dict[str, torch.Tensor]:
        """Convert SC2 action to model target format."""
        action_id = action.function
        
        # Create target tensors
        targets = {
            'action_type': torch.tensor(action_id, dtype=torch.long),
            'delay': torch.tensor(0.0, dtype=torch.float32),  # Default delay
            'queued': torch.tensor(float(action.queue), dtype=torch.float32)
        }
        
        # Handle spatial actions
        if actions.FUNCTIONS[action_id].args and 'screen' in str(actions.FUNCTIONS[action_id].args):
            spatial_target = torch.zeros(self.config.SPATIAL_SIZE)
            x, y = action.arguments[0]
            spatial_target[y, x] = 1.0
            targets['target_location'] = spatial_target
            
        # Handle unit selection
        if hasattr(action, 'selected_units'):
            unit_mask = torch.zeros(self.config.MAX_ENTITIES)
            for unit_id in action.selected_units:
                if unit_id < self.config.MAX_ENTITIES:
                    unit_mask[unit_id] = 1.0
            targets['selected_units'] = unit_mask
            
        return targets 