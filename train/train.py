import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, Optional
import wandb

from ..models.alphastar import AlphaStar
from ..data.datamodule import SC2DataModule
from ..utils.checkpointing import ModelCheckpoint
from ..utils.logger import WandbLogger

class AlphaStarTrainer:
    def __init__(self, 
                 config,
                 wandb_project: str = "alphastar",
                 wandb_entity: Optional[str] = None,
                 wandb_run_name: Optional[str] = None):
        """
        Initialize AlphaStar trainer.
        
        Args:
            config: Training configuration
            wandb_project: W&B project name
            wandb_entity: W&B username or team name
            wandb_run_name: Name for this run
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model = AlphaStar(config)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # Setup data
        self.datamodule = SC2DataModule(config)
        self.datamodule.setup()
        
        # Setup logging and checkpointing
        self.wandb_logger = WandbLogger(
            config=config,
            project_name=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name
        )
        self.checkpoint = ModelCheckpoint(config.LOG_DIR)
        
        # Log model architecture
        self.wandb_logger.log_model_graph(self.model)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        if resume_from:
            checkpoint = self.checkpoint.load(
                resume_from,
                self.model,
                self.optimizer
            )
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['step']
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        try:
            for epoch in range(self.epoch, self.config.NUM_EPOCHS):
                self.epoch = epoch
                train_metrics = self._train_epoch(device)
                
                # Log training metrics
                self.wandb_logger.log_training_step(
                    self.global_step,
                    train_metrics
                )
                
                # Validation
                val_metrics = self._validate(device)
                self.wandb_logger.log_evaluation(
                    self.global_step,
                    val_metrics
                )
                
                # Save checkpoint
                is_best = val_metrics['value_loss'] < self.best_metric
                if is_best:
                    self.best_metric = val_metrics['value_loss']
                
                checkpoint_path = self.checkpoint.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.global_step,
                    val_metrics,
                    is_best
                )
                
                # Log checkpoint as artifact
                self.wandb_logger.log_media(
                    self.global_step,
                    {'checkpoint': wandb.Artifact(
                        name=f"checkpoint-{self.global_step}",
                        type="model",
                        metadata=val_metrics
                    )}
                )
        finally:
            # Ensure wandb run is properly closed
            self.wandb_logger.finish()
    
    def evaluate(self, checkpoint_path: str, num_games: int = 10):
        """Evaluate model performance."""
        self.checkpoint.load(checkpoint_path, self.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        # Run evaluation games and collect metrics
        metrics = self._run_evaluation_games(num_games)
        
        # Log evaluation metrics
        self.wandb_logger.log_evaluation(
            self.global_step,
            metrics
        )
        
        return metrics
    
    def _train_epoch(self, device: torch.device) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = {}
        
        for batch in self.datamodule.train_dataloader():
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch['model_input'])
            losses = self._compute_losses(outputs, batch['target'])
            total_loss = sum(losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )
            self.optimizer.step()
            
            # Update metrics
            for k, v in losses.items():
                metrics[k] = metrics.get(k, 0) + v.item()
            
            self.global_step += 1
        
        # Average metrics
        metrics = {k: v / len(self.datamodule.train_dataloader())
                  for k, v in metrics.items()}
        
        return metrics
    
    def _validate(self, device: torch.device) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_loader = self.datamodule.val_dataloader()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch['model_input'])
                
                # Calculate losses
                losses = self._compute_losses(outputs, batch['target'])
                
                # Accumulate losses
                for k, v in losses.items():
                    total_losses[k] = total_losses.get(k, 0) + v.item()
                    
                num_batches += 1
                
        # Average losses
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses."""
        losses = {}
        
        # Action type loss
        losses['action_type'] = nn.CrossEntropyLoss()(
            outputs['action_type'],
            targets['action_type']
        )
        
        # Delay loss
        losses['delay'] = nn.MSELoss()(
            outputs['delay'],
            targets['delay']
        )
        
        # Queued loss
        losses['queued'] = nn.BCELoss()(
            outputs['queued'],
            targets['queued']
        )
        
        # Selected units loss
        losses['selected_units'] = nn.CrossEntropyLoss()(
            outputs['selected_units'],
            targets['selected_units']
        )
        
        # Target unit loss
        losses['target_unit'] = nn.CrossEntropyLoss()(
            outputs['target_unit'],
            targets['target_unit']
        )
        
        # Target location loss
        losses['target_location'] = nn.MSELoss()(
            outputs['target_location'],
            targets['target_location']
        )
        
        # Value loss
        losses['value'] = self.config.VALUE_LOSS_COEF * nn.MSELoss()(
            outputs['value'],
            targets['value']
        )
        
        return losses 