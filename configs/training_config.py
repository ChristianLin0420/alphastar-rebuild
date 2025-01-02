class TrainingConfig:
    """Configuration for AlphaStar training."""
    
    # Model architecture
    SCALAR_INPUT_DIM = 9  # Number of scalar features
    ENTITY_INPUT_DIM = 10  # Number of entity features
    SPATIAL_INPUT_CHANNELS = 32
    NUM_ACTIONS = 1000  # Total number of possible actions
    SPATIAL_SIZE = (128, 128)
    MAX_ENTITIES = 512
    
    # Network dimensions
    LSTM_HIDDEN_DIM = 512
    LSTM_NUM_LAYERS = 3
    TRANSFORMER_NUM_HEADS = 8
    TRANSFORMER_NUM_LAYERS = 3
    
    # Training hyperparameters
    BATCH_SIZE = 512
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 100
    GRADIENT_CLIP = 1.0
    
    # PPO hyperparameters
    PPO_EPOCHS = 3
    PPO_CLIP = 0.2
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # GAE parameter
    
    # Environment settings
    NUM_ENVS = 32
    STEPS_PER_UPDATE = 128
    
    # Evaluation settings
    EVAL_FREQUENCY = 1000  # Steps between evaluations
    EVAL_EPISODES = 10
    
    # Logging settings
    LOG_DIR = "logs"
    SAVE_FREQUENCY = 5000
    
    # Data settings
    REPLAY_BUFFER_SIZE = 100000
    NUM_WORKERS = 4
    
    # SC2 specific settings
    RACE = "terran"
    OPPONENT_RACE = "random"
    DIFFICULTY = "very_easy"
    MAP_NAME = "Simple64"
    STEP_MUL = 8 
    
    # Add wandb settings
    WANDB_PROJECT = "alphastar"
    WANDB_ENTITY = None  # Set to your username/team
    WANDB_WATCH_MODEL = True  # Whether to log model gradients 