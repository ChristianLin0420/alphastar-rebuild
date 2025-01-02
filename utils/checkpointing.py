import torch
import os
from typing import Dict, Optional
from ..models.alphastar import AlphaStar
import wandb

class ModelCheckpoint:
    """
    Handles model saving and loading.
    
    Attributes:
        save_dir (str): Directory to save checkpoints
        prefix (str): Prefix for checkpoint files
    """
    
    def __init__(self, save_dir: str, prefix: str = "alphastar"):
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, 
             model: AlphaStar,
             optimizer: torch.optim.Optimizer,
             epoch: int,
             step: int,
             metrics: Dict[str, float],
             is_best: bool = False) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: AlphaStar model
            optimizer: Optimizer
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'config': model.config
        }
        
        # Save regular checkpoint
        filename = f"{self.prefix}_step_{step}.pt"
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.save_dir, f"{self.prefix}_best.pt")
            torch.save(checkpoint, best_path)
            
            # Log best model as wandb artifact
            artifact = wandb.Artifact(
                name="best_model",
                type="model",
                metadata=metrics
            )
            artifact.add_file(best_path)
            wandb.log_artifact(artifact)
        
        return path
    
    def load(self, 
             path: str,
             model: Optional[AlphaStar] = None,
             optimizer: Optional[torch.optim.Optimizer] = None
            ) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            model: Optional model to load state into
            optimizer: Optional optimizer to load state into
            
        Returns:
            Dictionary containing checkpoint data
        """
        checkpoint = torch.load(path)
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint 