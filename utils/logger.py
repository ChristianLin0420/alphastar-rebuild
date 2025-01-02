import wandb
import logging
from typing import Dict, Any, Optional

class WandbLogger:
    """
    Logger for tracking experiments using Weights & Biases.
    
    Attributes:
        project_name (str): Name of the W&B project
        entity (str): W&B username or team name
        run_name (str): Name of the current run
    """
    
    def __init__(self, 
                 config: Any,
                 project_name: str = "alphastar",
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None):
        """
        Initialize W&B logger.
        
        Args:
            config: Training configuration
            project_name: W&B project name
            entity: W&B username or team name
            run_name: Name for this run
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize W&B
        wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config=vars(config),
            reinit=True
        )
        
        self.logger.info(f"Initialized W&B run: {wandb.run.name}")
    
    def log_training_step(self, step: int, metrics: Dict[str, float]):
        """Log training metrics for a single step."""
        wandb.log({"train/" + k: v for k, v in metrics.items()}, step=step)
    
    def log_evaluation(self, step: int, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        wandb.log({"eval/" + k: v for k, v in metrics.items()}, step=step)
    
    def log_model_graph(self, model):
        """Log model architecture."""
        wandb.watch(model, log="all")
    
    def log_media(self, step: int, media_dict: Dict[str, Any]):
        """Log media files (images, videos, etc.)."""
        wandb.log(media_dict, step=step)
    
    def finish(self):
        """Finish logging."""
        wandb.finish() 