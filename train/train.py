import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, List, Tuple
import wandb
from collections import deque
import logging

from ..models.alphastar import AlphaStar
from ..data.datamodule import SC2DataModule
from ..utils.checkpointing import ModelCheckpoint
from ..utils.logger import WandbLogger
from ..env.sc2_env import SC2Environment
from ..utils.replay_buffer import PrioritizedReplayBuffer

class AlphaStarTrainer:
    """
    Trainer class implementing both supervised learning and reinforcement learning for AlphaStar.
    
    Training Phases:
    1. Supervised Learning (SL):
       - Learn from human demonstrations
       - Behavioral cloning of macro-actions
       - Auxiliary tasks for better representations
       
    2. Reinforcement Learning (RL):
       - League training with different roles
       - Self-play between agents
       - Population-based training
       
    Features:
    - Multi-agent training coordination
    - Prioritized experience replay
    - League matchmaking
    - Population-based training
    - Hierarchical learning of strategies
    """
    
    def __init__(self,
                 config: Dict,
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
        
        # Initialize model and optimizer
        self.model = AlphaStar(config)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training_params']['learning_rate'],
            betas=config['training_params']['adam_betas']
        )
        
        # Setup data and environment
        self.datamodule = SC2DataModule(config)
        self.env = SC2Environment(config['env_params'])
        
        # Setup experience replay for RL
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config['rl_params']['replay_buffer_size'],
            alpha=config['rl_params']['priority_alpha']
        )
        
        # Setup logging and checkpointing
        self.wandb_logger = WandbLogger(
            config=config,
            project_name=wandb_project,
            entity=wandb_entity,
            run_name=wandb_run_name
        )
        self.checkpoint = ModelCheckpoint(config['training_params']['log_dir'])
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_win_rate = 0.0
        
        # League training state
        if config['training_params']['use_league']:
            self.setup_league()
    
    def setup_league(self):
        """Setup league training components."""
        self.league_agents = []
        self.matchmaking_ratings = {}
        self.win_rates = {}
    
    def supervised_training(self, num_epochs: int):
        """
        Supervised learning phase using human demonstrations.
        
        Args:
            num_epochs: Number of epochs to train
        """
        self.model.train()
        self.datamodule.setup(stage='fit')
        
        for epoch in range(num_epochs):
            epoch_metrics = self._train_supervised_epoch()
            
            # Validation
            val_metrics = self._validate_supervised()
            
            # Log metrics
            self.wandb_logger.log_metrics({
                'epoch': epoch,
                'supervised/train': epoch_metrics,
                'supervised/val': val_metrics
            })
            
            # Save checkpoint
            is_best = val_metrics['win_rate'] > self.best_win_rate
            if is_best:
                self.best_win_rate = val_metrics['win_rate']
            
            self.checkpoint.save(
                self.model,
                self.optimizer,
                epoch,
                self.global_step,
                val_metrics,
                is_best
            )
    
    def _train_supervised_epoch(self) -> Dict[str, float]:
        """Train one epoch with supervised learning."""
        metrics = {
            'action_loss': 0.0,
            'value_loss': 0.0,
            'auxiliary_loss': 0.0,
            'total_loss': 0.0
        }
        
        for batch in self.datamodule.train_dataloader():
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                inputs=batch['observations'],
                masks=batch.get('masks'),
                mode='supervised'
            )
            
            # Calculate losses
            losses = self._compute_supervised_losses(outputs, batch['targets'])
            total_loss = sum(losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training_params']['max_grad_norm']
            )
            
            self.optimizer.step()
            
            # Update metrics
            for k, v in losses.items():
                metrics[k] += v.item()
            metrics['total_loss'] += total_loss.item()
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.datamodule.train_dataloader())
        return {k: v / num_batches for k, v in metrics.items()}
    
    def reinforcement_training(self, num_steps: int):
        """
        Reinforcement learning phase with league training.
        
        Args:
            num_steps: Number of environment steps to train
        """
        self.model.train()
        steps_done = 0
        episode_rewards = deque(maxlen=100)
        
        while steps_done < num_steps:
            # Sample opponent from league
            if self.config['training_params']['use_league']:
                opponent = self._sample_league_opponent()
            else:
                opponent = None
            
            # Run episode
            episode_data = self._run_rl_episode(opponent)
            steps_done += len(episode_data['rewards'])
            episode_rewards.append(sum(episode_data['rewards']))
            
            # Store in replay buffer
            self.replay_buffer.add(episode_data)
            
            # Training update
            if len(self.replay_buffer) >= self.config['rl_params']['batch_size']:
                metrics = self._update_rl()
                
                # Log metrics
                self.wandb_logger.log_metrics({
                    'rl/policy_loss': metrics['policy_loss'],
                    'rl/value_loss': metrics['value_loss'],
                    'rl/entropy': metrics['entropy'],
                    'rl/avg_reward': np.mean(episode_rewards)
                })
            
            # League update
            if self.config['training_params']['use_league']:
                self._update_league_ratings(episode_data)
    
    def _run_rl_episode(self, opponent=None) -> Dict:
        """Run a single RL episode."""
        state = self.env.reset()
        done = False
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                outputs = self.model(
                    inputs={'observation': state},
                    mode='rl'
                )
            
            action = outputs['actions']
            value = outputs['baseline_value']
            log_prob = outputs['action_log_probs']
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['values'].append(value)
            episode_data['log_probs'].append(log_prob)
            
            state = next_state
        
        return episode_data
    
    def _update_rl(self) -> Dict[str, float]:
        """Perform RL update using PPO."""
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(
            self.config['rl_params']['batch_size']
        )
        
        # Calculate advantages
        advantages = self._compute_advantages(
            batch['rewards'],
            batch['values']
        )
        
        # PPO update
        for _ in range(self.config['rl_params']['ppo_epochs']):
            # Forward pass
            outputs = self.model(
                inputs=batch['states'],
                mode='rl'
            )
            
            # Calculate losses
            losses = self._compute_rl_losses(
                outputs,
                batch,
                advantages
            )
            
            total_loss = (
                losses['policy_loss'] +
                self.config['rl_params']['value_loss_coef'] * losses['value_loss'] -
                self.config['rl_params']['entropy_coef'] * losses['entropy']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['rl_params']['max_grad_norm']
            )
            self.optimizer.step()
        
        return losses
    
    def _compute_supervised_losses(self,
                                 outputs: Dict[str, torch.Tensor],
                                 targets: Dict[str, torch.Tensor]
                                ) -> Dict[str, torch.Tensor]:
        """Compute supervised learning losses."""
        losses = {}
        
        # Action prediction losses
        losses['action_loss'] = 0
        for key in ['action_type', 'delay', 'queued', 'selected_units',
                   'target_unit', 'target_location']:
            if key in outputs and key in targets:
                if key in ['delay', 'queued']:
                    losses['action_loss'] += nn.BCEWithLogitsLoss()(
                        outputs[key], targets[key]
                    )
                else:
                    losses['action_loss'] += nn.CrossEntropyLoss()(
                        outputs[key], targets[key]
                    )
        
        # Value prediction loss
        losses['value_loss'] = nn.MSELoss()(
            outputs['supervised_value'],
            targets['value']
        )
        
        # Auxiliary losses
        if 'auxiliary' in outputs and 'auxiliary' in targets:
            losses['auxiliary_loss'] = sum(
                nn.MSELoss()(outputs['auxiliary'][k], targets['auxiliary'][k])
                for k in outputs['auxiliary']
            )
        
        return losses
    
    def _compute_rl_losses(self,
                          outputs: Dict[str, torch.Tensor],
                          batch: Dict[str, torch.Tensor],
                          advantages: torch.Tensor
                         ) -> Dict[str, torch.Tensor]:
        """Compute reinforcement learning losses."""
        # Policy loss (PPO)
        ratio = torch.exp(
            outputs['action_log_probs'] - batch['log_probs']
        )
        
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(
            ratio,
            1 - self.config['rl_params']['clip_epsilon'],
            1 + self.config['rl_params']['clip_epsilon']
        ) * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Value loss
        value_pred = outputs['baseline_value']
        value_target = batch['returns']
        value_loss = nn.MSELoss()(value_pred, value_target)
        
        # Entropy loss for exploration
        entropy = -(outputs['action_probs'] * outputs['action_log_probs']).sum(-1).mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
    
    def _compute_advantages(self,
                          rewards: torch.Tensor,
                          values: torch.Tensor
                         ) -> torch.Tensor:
        """Compute advantages using GAE."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = (
                rewards[t] +
                self.config['rl_params']['gamma'] * next_value -
                values[t]
            )
            
            advantages[t] = last_gae = (
                delta +
                self.config['rl_params']['gamma'] *
                self.config['rl_params']['gae_lambda'] *
                last_gae
            )
        
        return advantages
    
    def _sample_league_opponent(self):
        """Sample opponent from league using matchmaking."""
        if not self.league_agents:
            return None
            
        # Calculate selection probabilities based on ratings
        ratings = np.array([
            self.matchmaking_ratings.get(agent.id, 1500)
            for agent in self.league_agents
        ])
        
        # Add exploration bonus for less played opponents
        games_played = np.array([
            self.win_rates[agent.id]['games']
            for agent in self.league_agents
        ])
        exploration_bonus = 1.0 / (games_played + 1)
        
        # Combine ratings and exploration
        scores = ratings + self.config['league_params']['exploration_weight'] * exploration_bonus
        probs = torch.softmax(torch.tensor(scores), dim=0)
        
        # Sample opponent
        idx = torch.multinomial(probs, 1).item()
        return self.league_agents[idx]
    
    def _update_league_ratings(self, episode_data: Dict):
        """Update matchmaking ratings based on game outcome."""
        if not episode_data.get('opponent_id'):
            return
            
        # Extract outcome
        won = episode_data['reward'] > 0
        opponent_id = episode_data['opponent_id']
        
        # Update Elo ratings
        k_factor = self.config['league_params']['elo_k_factor']
        agent_rating = self.matchmaking_ratings.get(self.model.id, 1500)
        opponent_rating = self.matchmaking_ratings.get(opponent_id, 1500)
        
        # Expected scores
        ea = 1 / (1 + 10**((opponent_rating - agent_rating) / 400))
        eb = 1 - ea
        
        # Update ratings
        if won:
            agent_rating += k_factor * (1 - ea)
            opponent_rating += k_factor * (0 - eb)
        else:
            agent_rating += k_factor * (0 - ea)
            opponent_rating += k_factor * (1 - eb)
        
        # Store updated ratings
        self.matchmaking_ratings[self.model.id] = agent_rating
        self.matchmaking_ratings[opponent_id] = opponent_rating
        
        # Update win rates
        if self.model.id not in self.win_rates:
            self.win_rates[self.model.id] = {'wins': 0, 'games': 0}
        if opponent_id not in self.win_rates:
            self.win_rates[opponent_id] = {'wins': 0, 'games': 0}
        
        self.win_rates[self.model.id]['games'] += 1
        self.win_rates[opponent_id]['games'] += 1
        if won:
            self.win_rates[self.model.id]['wins'] += 1 