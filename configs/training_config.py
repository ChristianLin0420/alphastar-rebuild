"""
Training configuration for AlphaStar.
"""

class TrainingConfig:
    """Training configuration for AlphaStar."""
    
    def __init__(self):
        # Model dimensions
        self.SCALAR_INPUT_DIM = 32
        self.ENTITY_INPUT_DIM = 64
        self.SPATIAL_INPUT_DIM = 16
        self.MAX_ENTITIES = 100
        self.SPATIAL_SIZE = (64, 64)
        
        # Network parameters
        self.SCALAR_ENCODER_DIMS = [64, 32]
        self.ENCODER_OUTPUT_DIM = 32
        self.TRANSFORMER_DIM = 64
        self.TRANSFORMER_HEADS = 2
        self.TRANSFORMER_LAYERS = 2
        self.DROPOUT = 0.1
        self.SPATIAL_CHANNELS = 16
        self.NUM_RES_BLOCKS = 2
        self.LSTM_DIM = 128
        self.LSTM_LAYERS = 2
        
        # Action space
        self.NUM_ACTIONS = 100
        self.BUILD_ORDER_SIZE = 20
        self.UNIT_TYPE_SIZE = 32
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-4
        self.ADAM_BETAS = (0.9, 0.999)
        self.MAX_GRAD_NORM = 10.0
        self.USE_AUXILIARY = True
        self.USE_LEAGUE = True
        self.LOG_DIR = 'tests/test_logs'
        self.SL_BATCH_SIZE = 32
        self.SL_NUM_EPOCHS = 2
        self.SL_WARMUP_STEPS = 100
        self.NUM_WORKERS = 2
        self.PIN_MEMORY = True 