import argparse
import logging
import wandb
from pathlib import Path
from train.train import AlphaStarTrainer
from configs.training_config import TrainingConfig

def setup_logging():
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AlphaStar model')
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    parser.add_argument(
        '--evaluate',
        type=str,
        help='Path to model checkpoint for evaluation'
    )
    parser.add_argument(
        '--num-games',
        type=int,
        default=10,
        help='Number of games to evaluate'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='alphastar',
        help='W&B project name'
    )
    parser.add_argument(
        '--wandb-entity',
        type=str,
        help='W&B username or team name'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        help='W&B run name'
    )
    return parser.parse_args()

def main():
    """Main training entry point."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load config
    config = TrainingConfig()
    if args.config:
        logger.info(f"Loading custom config from {args.config}")
        # Implement custom config loading here
        pass
    
    # Initialize wandb
    logger.info("Initializing W&B logging")
    trainer = AlphaStarTrainer(
        config,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name
    )
    
    try:
        if args.evaluate:
            logger.info(f"Evaluating model from checkpoint: {args.evaluate}")
            trainer.evaluate(
                checkpoint_path=args.evaluate,
                num_games=args.num_games
            )
        else:
            if args.resume:
                logger.info(f"Resuming training from checkpoint: {args.resume}")
            trainer.train(resume_from=args.resume)
    finally:
        # Ensure wandb run is properly closed
        wandb.finish()

if __name__ == '__main__':
    main() 