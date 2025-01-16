#!/usr/bin/env python3
"""
Main training script for AlphaStar.
"""

import os
import sys
import argparse
from pathlib import Path
import logging

import torch
import wandb

from configs.config import load_config, print_config
from train.train import AlphaStarTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train AlphaStar agent')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--mode', type=str, default='supervised',
                       choices=['supervised', 'reinforcement'],
                       help='Training mode')
    parser.add_argument('--wandb-project', type=str, default='alphastar',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str,
                       help='Weights & Biases username/team name')
    parser.add_argument('--wandb-run-name', type=str,
                       help='Weights & Biases run name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    args = parse_args()
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Load and validate configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    print_config(config)
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize trainer
    trainer = AlphaStarTrainer(
        config=config,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )
    
    # Move model to device
    trainer.model = trainer.model.to(device)
    
    # Resume from checkpoint if specified
    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.checkpoint.load(
            args.checkpoint,
            trainer.model,
            trainer.optimizer
        )
    
    # Start training
    logger.info(f"Starting {args.mode} training")
    try:
        if args.mode == 'supervised':
            trainer.supervised_training(
                num_epochs=config['training_params']['sl_num_epochs']
            )
        else:
            trainer.reinforcement_training(
                num_steps=config['rl_params']['total_timesteps']
            )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("Training failed")
        raise
    finally:
        # Ensure wandb run is properly closed
        wandb.finish()

if __name__ == '__main__':
    main() 