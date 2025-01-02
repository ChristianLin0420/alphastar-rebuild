import torch
from typing import Dict, Tuple, List, Optional

def validate_scalar_input(scalar_features: Dict[str, float]) -> torch.Tensor:
    """
    Validate and convert scalar features to tensor format.
    
    Args:
        scalar_features (Dict[str, float]): Dictionary of scalar features
        
    Returns:
        torch.Tensor: Validated and formatted tensor
        
    Raises:
        ValueError: If required features are missing or invalid
    """
    required_features = [
        'minerals', 'vespene', 'food_used', 'food_cap',
        'army_count', 'worker_count'
    ]
    
    # Check for required features
    missing_features = [f for f in required_features if f not in scalar_features]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for valid numeric values
    for feature, value in scalar_features.items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Feature {feature} has invalid value {value}. "
                f"Expected numeric value."
            )
    
    # Convert to tensor
    features = [scalar_features[f] for f in required_features]
    return torch.tensor(features, dtype=torch.float32)

def validate_model_inputs(
    scalar_input: torch.Tensor,
    entity_input: torch.Tensor,
    spatial_input: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validate model inputs and ensure they have correct shapes and types.
    
    Args:
        scalar_input (torch.Tensor): Scalar features
        entity_input (torch.Tensor): Entity features
        spatial_input (torch.Tensor): Spatial features
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Validated inputs
        
    Raises:
        ValueError: If inputs have incorrect shapes or types
    """
    # Validate scalar input
    if not isinstance(scalar_input, torch.Tensor):
        raise ValueError("scalar_input must be a torch.Tensor")
    if scalar_input.dim() != 2:
        raise ValueError(f"scalar_input must be 2D, got shape {scalar_input.shape}")
    
    # Validate entity input
    if not isinstance(entity_input, torch.Tensor):
        raise ValueError("entity_input must be a torch.Tensor")
    if entity_input.dim() != 3:
        raise ValueError(f"entity_input must be 3D, got shape {entity_input.shape}")
    
    # Validate spatial input
    if not isinstance(spatial_input, torch.Tensor):
        raise ValueError("spatial_input must be a torch.Tensor")
    if spatial_input.dim() != 4:
        raise ValueError(f"spatial_input must be 4D, got shape {spatial_input.shape}")
    
    return scalar_input, entity_input, spatial_input 

def validate_entity_input(entities: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validate and convert entity features to tensor format with padding.
    
    Args:
        entities (List[Dict]): List of entity dictionaries with features
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Entity features and mask tensors
        
    Raises:
        ValueError: If entity features are invalid or missing required fields
    """
    required_fields = ['unit_type', 'alliance', 'health', 'shield', 'energy', 'x', 'y']
    max_entities = 512
    feature_dim = len(required_fields)
    
    # Initialize tensors
    features = torch.zeros(max_entities, feature_dim)
    mask = torch.zeros(max_entities, dtype=torch.bool)
    
    # Validate and fill features
    for i, entity in enumerate(entities[:max_entities]):
        # Check required fields
        missing_fields = [f for f in required_fields if f not in entity]
        if missing_fields:
            raise ValueError(f"Entity {i} missing required fields: {missing_fields}")
        
        # Validate numeric values
        for field in required_fields:
            if not isinstance(entity[field], (int, float)):
                raise ValueError(
                    f"Entity {i}, field {field} has invalid value {entity[field]}. "
                    f"Expected numeric value."
                )
        
        # Fill features
        features[i] = torch.tensor([entity[f] for f in required_fields])
        mask[i] = True
    
    return features, mask 

def validate_spatial_input(minimap: torch.Tensor, expected_size: tuple) -> torch.Tensor:
    """
    Validate and preprocess spatial input.
    
    Args:
        minimap (torch.Tensor): Spatial features tensor
        expected_size (tuple): Expected spatial dimensions (height, width)
        
    Returns:
        torch.Tensor: Validated and preprocessed spatial features
        
    Raises:
        ValueError: If input tensor has incorrect shape or values
    """
    if not isinstance(minimap, torch.Tensor):
        raise ValueError("Minimap must be a torch.Tensor")
        
    if minimap.dim() != 4:
        raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {minimap.shape}")
        
    if minimap.shape[-2:] != expected_size:
        raise ValueError(f"Expected spatial size {expected_size}, got {minimap.shape[-2:]}")
        
    # Normalize values to [0, 1] if not already
    if minimap.min() < 0 or minimap.max() > 1:
        minimap = (minimap - minimap.min()) / (minimap.max() - minimap.min())
        
    return minimap 

def validate_core_input(
    x: torch.Tensor,
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    num_layers: int = 3,
    hidden_dim: int = 512
) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Validate inputs for the core LSTM module.
    
    Args:
        x (torch.Tensor): Input tensor
        hidden (Optional[Tuple[torch.Tensor, torch.Tensor]]): Hidden state tuple
        num_layers (int): Number of LSTM layers
        hidden_dim (int): Hidden state dimension
        
    Returns:
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]: 
            Validated input and hidden state
            
    Raises:
        ValueError: If inputs have incorrect shapes or types
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")
        
    if x.dim() != 3:
        raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
        
    if hidden is not None:
        if not isinstance(hidden, tuple) or len(hidden) != 2:
            raise ValueError("Hidden state must be a tuple of (h_0, c_0)")
            
        h_0, c_0 = hidden
        expected_shape = (num_layers, x.size(0), hidden_dim)
        
        if h_0.shape != expected_shape or c_0.shape != expected_shape:
            raise ValueError(
                f"Hidden state tensors must have shape {expected_shape}, "
                f"got {h_0.shape} and {c_0.shape}"
            )
    
    return x, hidden 