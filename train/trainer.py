import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..configs.training_config import TrainingConfig
from ..train.replay_buffer import ReplayBuffer
from ..utils.logger import AlphaStarLogger
from ..eval.evaluator import AlphaStarEvaluator

class AlphaStarTrainer:
    def __init__(self, model, config=TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.logger = AlphaStarLogger(config)
        self.evaluator = AlphaStarEvaluator(model, config)
        self.global_step = 0
        
    def compute_loss(self, outputs, targets):
        losses = {
            'action_type': nn.CrossEntropyLoss()(outputs['action_type'], targets['action_type']),
            'delay': nn.MSELoss()(outputs['delay'], targets['delay']),
            'queued': nn.BCELoss()(outputs['queued'], targets['queued']),
            'selected_units': nn.CrossEntropyLoss()(outputs['selected_units'], targets['selected_units']),
            'target_unit': nn.CrossEntropyLoss()(outputs['target_unit'], targets['target_unit']),
            'target_location': nn.MSELoss()(outputs['target_location'], targets['target_location']),
            'value': nn.MSELoss()(outputs['value'], targets['value'])
        }
        
        return sum(losses.values()), losses
    
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            batch['scalar_input'],
            batch['entity_input'],
            batch['spatial_input'],
            batch.get('mask', None)
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(outputs, batch['targets'])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
        self.optimizer.step()
        
        return loss_dict 
    
    def train_ppo(self, env, num_episodes: int):
        """Train using PPO algorithm."""
        replay_buffer = ReplayBuffer(self.config.BATCH_SIZE * 10)
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from model
                with torch.no_grad():
                    outputs = self.model(**state)
                
                # Execute action in environment
                next_state, reward, done, _ = env.step(outputs)
                
                # Store transition
                replay_buffer.push(state, outputs, reward, next_state, done)
                
                # Update state and episode reward
                state = next_state
                episode_reward += reward
                
                # PPO update
                if len(replay_buffer) >= self.config.BATCH_SIZE:
                    losses = self._ppo_update(replay_buffer)
                    self.logger.log_training_step(self.global_step, losses)
                    self.global_step += 1
                
                # Periodic evaluation
                if self.global_step % self.config.EVAL_FREQUENCY == 0:
                    metrics = self.evaluator.evaluate()
                    self.logger.log_evaluation(self.global_step, metrics)
                    self.logger.save_model(self.model, self.global_step)
            
            print(f"Episode {episode}, Reward: {episode_reward}")
        
        self.logger.close()
    
    def _ppo_update(self, replay_buffer):
        """Update model using PPO."""
        states, old_actions, rewards, next_states, dones = \
            replay_buffer.sample(self.config.BATCH_SIZE)
        
        # Compute advantages
        with torch.no_grad():
            next_values = self.model(**next_states)['value']
            advantages = rewards + (1 - dones) * self.config.GAMMA * next_values
        
        # PPO iterations
        for _ in range(self.config.PPO_EPOCHS):
            # Get current action probabilities and values
            outputs = self.model(**states)
            
            # Compute PPO loss
            ratio = torch.exp(outputs['action_type'] - old_actions['action_type'])
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.PPO_CLIP,
                               1 + self.config.PPO_CLIP) * advantages
            
            # Compute losses
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.config.VALUE_LOSS_COEF * \
                        (outputs['value'] - rewards).pow(2).mean()
            entropy_loss = -self.config.ENTROPY_COEF * \
                          torch.distributions.Categorical(
                              outputs['action_type']).entropy().mean()
            
            # Total loss
            loss = action_loss + value_loss + entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.GRADIENT_CLIP)
            self.optimizer.step() 