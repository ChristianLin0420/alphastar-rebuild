"""
Configuration utilities for loading and validating YAML configs.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    _validate_config(config)
    
    return config

def print_config(config: Dict, indent: int = 0) -> None:
    """
    Print configuration dictionary in a nicely formatted hierarchical structure.
    
    Args:
        config: Configuration dictionary to print
        indent: Current indentation level (used for recursion)
    """
    for key, value in config.items():
        # Print key with proper indentation
        print("  " * indent + f"{key}:")
        
        # Recursively print nested dictionaries
        if isinstance(value, dict):
            print_config(value, indent + 1)
        # Print lists in a readable format    
        elif isinstance(value, list):
            if all(not isinstance(x, (dict, list)) for x in value):
                print("  " * (indent + 1) + f"{value}")
            else:
                for item in value:
                    if isinstance(item, dict):
                        print_config(item, indent + 1)
                    else:
                        print("  " * (indent + 1) + f"- {item}")
        # Print other values directly
        else:
            print("  " * (indent + 1) + f"{value}")


def _validate_config(config: Dict) -> None:
    """
    Validate configuration dictionary has all required sections.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If required sections are missing or invalid
    """
    required_sections = {
        'input_dims': ['scalar', 'entity', 'spatial'],
        'network_params': [
            'scalar_encoder',
            'encoder_output',
            'transformer_dim',
            'transformer_heads', 
            'transformer_layers',
            'dropout',
            'spatial_channels',
            'num_res_blocks',
            'lstm_dim',
            'lstm_layers'
        ],
        'action_space': [
            'num_actions',
            'spatial_size',
            'build_order_size',
            'unit_type_size'
        ],
        'training_params': [
            'learning_rate',
            'adam_betas',
            'max_grad_norm',
            'use_auxiliary',
            'use_league',
            'log_dir',
            'sl_batch_size',
            'sl_num_epochs',
            'sl_warmup_steps',
            'num_workers',
            'pin_memory'
        ],
        'rl_params': [
            'gamma',
            'gae_lambda',
            'clip_epsilon',
            'value_loss_coef',
            'entropy_coef',
            'ppo_epochs',
            'batch_size',
            'max_grad_norm',
            'replay_buffer_size',
            'priority_alpha',
            'priority_beta',
            'priority_epsilon',
            'total_timesteps',
            'update_interval'
        ],
        'league_params': [
            'initial_rating',
            'elo_k_factor',
            'exploration_weight',
            'min_games_played',
            'population_size',
            'mutation_rate',
            'crossover_rate',
            'match_winrate_threshold',
            'required_games',
            'max_active_players'
        ],
        'env_params': [
            'game_steps_per_episode',
            'step_mul',
            'max_episode_steps',
            'use_feature_units',
            'use_raw_units',
            'use_unit_counts',
            'use_camera_position',
            'reward_scale',
            'reward_death_value',
            'reward_win',
            'reward_defeat',
            'reward_step',
            'action_space_config'
        ],
        'eval_params': [
            'eval_frequency',
            'num_eval_episodes',
            'save_replay',
            'eval_deterministic'
        ]
    }
    
    # Check all required sections exist
    for section, params in required_sections.items():
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
        
        # Check all required parameters exist in each section
        for param in params:
            if param not in config[section]:
                raise ValueError(f"Missing required parameter '{param}' in section '{section}'")
    
    # Validate specific parameter types and values
    _validate_parameter_values(config)

def _validate_parameter_values(config: Dict) -> None:
    """
    Validate specific parameter types and values.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If parameter values are invalid
    """
    # Validate network dimensions are positive
    for key in ['scalar', 'entity', 'spatial']:
        if config['input_dims'][key] <= 0:
            raise ValueError(f"Input dimension '{key}' must be positive")
    
    # Validate learning parameters are positive
    if config['training_params']['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    if not (0 <= config['network_params']['dropout'] <= 1):
        raise ValueError("Dropout rate must be between 0 and 1")
    
    # Validate RL parameters are in valid ranges
    rl_params = config['rl_params']
    if not (0 <= rl_params['gamma'] <= 1):
        raise ValueError("Discount factor (gamma) must be between 0 and 1")
    
    if not (0 <= rl_params['gae_lambda'] <= 1):
        raise ValueError("GAE lambda must be between 0 and 1")
    
    if rl_params['clip_epsilon'] <= 0:
        raise ValueError("PPO clip epsilon must be positive")
    
    # Validate buffer sizes are positive
    if rl_params['replay_buffer_size'] <= 0:
        raise ValueError("Replay buffer size must be positive")
    
    # Validate league parameters
    league_params = config['league_params']
    if league_params['population_size'] <= 0:
        raise ValueError("Population size must be positive")
    
    if not (0 <= league_params['mutation_rate'] <= 1):
        raise ValueError("Mutation rate must be between 0 and 1")
    
    if not (0 <= league_params['crossover_rate'] <= 1):
        raise ValueError("Crossover rate must be between 0 and 1")

def save_config(config: Dict, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False) 