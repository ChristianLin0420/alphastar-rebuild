"""
Configuration for AlphaStar training, including both supervised and reinforcement learning settings.
"""

def get_config():
    """Get the complete training configuration."""
    config = {
        # Input dimensions for different modalities
        'input_dims': {
            'scalar': 512,      # Game statistics, resources, etc.
            'entity': 256,      # Unit and building features
            'spatial': 64,      # Map features
        },
        
        # Network architecture parameters
        'network_params': {
            # Encoder parameters
            'scalar_encoder': [512, 256],  # Hidden layer dimensions
            'encoder_output': 256,         # Final encoder output dimension
            
            # Transformer parameters for entity encoding
            'transformer_dim': 256,
            'transformer_heads': 4,
            'transformer_layers': 3,
            'dropout': 0.1,
            
            # Spatial encoder parameters
            'spatial_channels': 64,
            'num_res_blocks': 4,
            
            # Core LSTM parameters
            'lstm_dim': 384,
            'lstm_layers': 3,
        },
        
        # Action space configuration
        'action_space': {
            'num_actions': 1000,          # Total number of possible actions
            'spatial_size': (256, 256),   # Resolution of spatial actions
            'build_order_size': 100,      # Number of build order predictions
            'unit_type_size': 256,        # Number of unit types
        },
        
        # Training parameters
        'training_params': {
            # General
            'learning_rate': 1e-4,
            'adam_betas': (0.9, 0.999),
            'max_grad_norm': 10.0,
            'use_auxiliary': True,
            'use_league': True,
            'log_dir': 'logs/alphastar',
            
            # Supervised learning
            'sl_batch_size': 512,
            'sl_num_epochs': 100,
            'sl_warmup_steps': 10000,
            
            # Data sampling
            'num_workers': 8,
            'pin_memory': True,
        },
        
        # Reinforcement learning parameters
        'rl_params': {
            # PPO parameters
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'ppo_epochs': 3,
            'batch_size': 2048,
            'max_grad_norm': 0.5,
            
            # Experience replay
            'replay_buffer_size': 100000,
            'priority_alpha': 0.6,
            'priority_beta': 0.4,
            'priority_epsilon': 1e-6,
            
            # Training schedule
            'total_timesteps': 10000000,
            'update_interval': 2048,
        },
        
        # League training parameters
        'league_params': {
            'initial_rating': 1500,
            'elo_k_factor': 32,
            'exploration_weight': 500,
            'min_games_played': 5,
            
            # Population parameters
            'population_size': 10,
            'mutation_rate': 0.1,
            'crossover_rate': 0.5,
            
            # Matchmaking
            'match_winrate_threshold': 0.3,
            'required_games': 20,
            'max_active_players': 5,
        },
        
        # Environment parameters
        'env_params': {
            'game_steps_per_episode': 48000,    # 20 minutes at 40 APM
            'step_mul': 8,                      # Number of game steps per agent step
            'random_seed': None,                # Random seed for environment
            'max_episode_steps': 10000,         # Maximum steps before episode timeout
            
            # Observation space
            'use_feature_units': True,
            'use_raw_units': False,
            'use_unit_counts': True,
            'use_camera_position': True,
            
            # Reward shaping
            'reward_scale': 1.0,
            'reward_death_value': -1.0,
            'reward_win': 1.0,
            'reward_defeat': -1.0,
            'reward_step': -0.0001,            # Small negative reward to encourage faster games
            
            # Action space
            'action_space_config': {
                'use_raw_actions': False,
                'use_feature_moves': True,
                'use_camera_moves': True,
                'max_selected_units': 64,
            },
        },
        
        # Evaluation parameters
        'eval_params': {
            'eval_frequency': 10000,        # Steps between evaluations
            'num_eval_episodes': 50,        # Number of episodes per evaluation
            'save_replay': True,            # Save game replays during evaluation
            'eval_deterministic': True,     # Use deterministic actions for evaluation
        },
    }
    
    return config 