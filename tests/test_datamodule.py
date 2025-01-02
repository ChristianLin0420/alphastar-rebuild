import pytest
import torch
from ..data.datamodule import SC2Dataset, SC2DataModule
from ..configs.training_config import TrainingConfig

def test_sc2_dataset():
    """Test SC2Dataset functionality."""
    config = TrainingConfig()
    datamodule = SC2DataModule(config)
    
    # Get dummy data
    obs, actions = datamodule._load_replays('train')
    dataset = SC2Dataset(obs, actions, config)
    
    # Test dataset size
    assert len(dataset) > 0
    
    # Test item retrieval
    item = dataset[0]
    assert isinstance(item, dict)
    assert 'model_input' in item
    assert 'target' in item
    
    # Test model input format
    model_input = item['model_input']
    assert all(key in model_input for key in 
              ['scalar_input', 'entity_input', 'spatial_input', 'entity_mask'])
    
    # Test target format
    target = item['target']
    assert 'action_type' in target
    assert isinstance(target['action_type'], torch.Tensor)

def test_sc2_datamodule():
    """Test SC2DataModule functionality."""
    config = TrainingConfig()
    datamodule = SC2DataModule(config)
    
    # Test setup
    datamodule.setup('fit')
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    
    # Test dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    assert isinstance(train_loader.dataset, SC2Dataset)
    assert isinstance(val_loader.dataset, SC2Dataset)
    
    # Test batch loading
    batch = next(iter(train_loader))
    assert isinstance(batch, dict)
    assert batch['model_input']['scalar_input'].shape[0] == config.BATCH_SIZE

@pytest.mark.parametrize("batch_size", [1, 16, 32])
def test_dataloader_batch_sizes(batch_size):
    """Test if dataloader handles different batch sizes correctly."""
    config = TrainingConfig()
    config.BATCH_SIZE = batch_size
    datamodule = SC2DataModule(config)
    
    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    
    batch = next(iter(train_loader))
    assert all(v.size(0) == batch_size 
              for v in batch['model_input'].values())

def test_data_shapes():
    """Test if data shapes match model requirements."""
    config = TrainingConfig()
    datamodule = SC2DataModule(config)
    datamodule.setup('fit')
    
    batch = next(iter(datamodule.train_dataloader()))
    model_input = batch['model_input']
    
    # Check scalar input shape
    assert model_input['scalar_input'].shape[1] == config.SCALAR_INPUT_DIM
    
    # Check entity input shape
    assert model_input['entity_input'].shape[1] == config.MAX_ENTITIES
    assert model_input['entity_input'].shape[2] == config.ENTITY_INPUT_DIM
    
    # Check spatial input shape
    assert model_input['spatial_input'].shape[1] == config.SPATIAL_INPUT_CHANNELS
    assert model_input['spatial_input'].shape[2:] == config.SPATIAL_SIZE 