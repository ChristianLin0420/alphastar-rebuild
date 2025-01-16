"""
Replay buffer implementations for experience replay in reinforcement learning.
Includes various sampling schemes:
1. Prioritized Experience Replay (PER)
2. Uniform Random Sampling
3. Episodic Replay
4. Hindsight Experience Replay (HER)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, namedtuple
import random
from abc import ABC, abstractmethod

# Define transition type for storing experiences
Transition = namedtuple('Transition', 
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])

class SumTree:
    """
    Binary sum tree data structure for efficient priority sampling.
    Used by PrioritizedReplayBuffer for O(log n) updates and sampling.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize sum tree with given capacity.
        
        Args:
            capacity: Maximum number of leaf nodes (experiences)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Binary tree array
        self.data = np.zeros(capacity, dtype=object)  # Experience storage
        self.size = 0  # Current number of elements
        self.next_idx = 0  # Next index to write to
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample index given a priority value s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def update(self, idx: int, priority: float):
        """Update priority at given index."""
        change = priority - self.tree[idx + self.capacity - 1]
        self.tree[idx + self.capacity - 1] = priority
        self._propagate(idx + self.capacity - 1, change)
    
    def add(self, priority: float, data: object):
        """Add new experience with given priority."""
        idx = self.next_idx
        self.data[idx] = data
        self.update(idx, priority)
        
        self.next_idx = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get experience based on priority value s."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])
    
    @property
    def total_priority(self) -> float:
        """Get sum of all priorities."""
        return self.tree[0]

class BaseReplayBuffer(ABC):
    """Abstract base class for replay buffers."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    @abstractmethod
    def add(self, *args, **kwargs):
        """Add experience to buffer."""
        pass
    
    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of experiences."""
        pass
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)

class UniformReplayBuffer(BaseReplayBuffer):
    """Simple replay buffer with uniform random sampling."""
    
    def add(self, state: torch.Tensor, action: torch.Tensor, 
            reward: float, next_state: torch.Tensor, 
            done: bool, info: Optional[Dict] = None):
        """Add transition to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Transition(
            state, action, reward, next_state, done, info
        )
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        # Transpose batch of transitions to batches for each element
        batch = Transition(*zip(*batch))
        
        return {
            'states': torch.stack(batch.state),
            'actions': torch.stack(batch.action),
            'rewards': torch.tensor(batch.reward),
            'next_states': torch.stack(batch.next_state),
            'dones': torch.tensor(batch.done),
            'infos': batch.info
        }

class PrioritizedReplayBuffer(BaseReplayBuffer):
    """
    Prioritized Experience Replay (PER) implementation.
    Uses sum tree for efficient priority-based sampling.
    
    Reference: https://arxiv.org/abs/1511.05952
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        Initialize PER buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Priority exponent (α), controls how much prioritization is used
            beta: Importance sampling exponent (β), corrects bias from prioritized sampling
            beta_increment: Increment β towards 1 over time
            epsilon: Small constant to ensure non-zero priorities
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, next_state: torch.Tensor,
            done: bool, info: Optional[Dict] = None):
        """Add transition with maximum priority."""
        transition = Transition(state, action, reward, next_state, done, info)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch based on priorities."""
        batch = []
        indices = []
        priorities = []
        total_priority = self.tree.total_priority
        
        # Calculate segment size for priority sampling
        segment = total_priority / batch_size
        
        # Update beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            # Retrieve experience and record sampling data
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        
        # Calculate importance sampling weights
        priorities = np.array(priorities) / total_priority
        weights = (self.capacity * priorities) ** -self.beta
        weights = weights / weights.max()  # Normalize weights
        
        # Transpose batch
        batch = Transition(*zip(*batch))
        
        return {
            'states': torch.stack(batch.state),
            'actions': torch.stack(batch.action),
            'rewards': torch.tensor(batch.reward),
            'next_states': torch.stack(batch.next_state),
            'dones': torch.tensor(batch.done),
            'infos': batch.info,
            'indices': torch.tensor(indices),
            'weights': torch.tensor(weights, dtype=torch.float32)
        }
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for given transitions."""
        for idx, priority in zip(indices, priorities):
            priority = (priority + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

class EpisodicReplayBuffer(BaseReplayBuffer):
    """
    Episodic replay buffer that stores complete episodes.
    Useful for algorithms that require full trajectory information.
    """
    
    def __init__(self, capacity: int, max_episode_length: int = 1000):
        """
        Initialize episodic buffer.
        
        Args:
            capacity: Maximum number of episodes to store
            max_episode_length: Maximum length of each episode
        """
        super().__init__(capacity)
        self.max_episode_length = max_episode_length
        self.current_episode = []
    
    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, next_state: torch.Tensor,
            done: bool, info: Optional[Dict] = None):
        """Add transition to current episode."""
        self.current_episode.append(
            Transition(state, action, reward, next_state, done, info)
        )
        
        if done or len(self.current_episode) >= self.max_episode_length:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = self.current_episode
            self.position = (self.position + 1) % self.capacity
            self.current_episode = []
    
    def sample(self, batch_size: int) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """Sample batch of episodes."""
        episodes = random.sample(self.buffer, batch_size)
        
        # Process each episode into tensor dictionary
        batch = []
        for episode in episodes:
            episode_data = Transition(*zip(*episode))
            batch.append({
                'states': torch.stack(episode_data.state),
                'actions': torch.stack(episode_data.action),
                'rewards': torch.tensor(episode_data.reward),
                'next_states': torch.stack(episode_data.next_state),
                'dones': torch.tensor(episode_data.done),
                'infos': episode_data.info
            })
        
        return batch

class HindsightReplayBuffer(BaseReplayBuffer):
    """
    Hindsight Experience Replay (HER) implementation.
    Augments failed experiences with alternative goals for better sample efficiency.
    
    Reference: https://arxiv.org/abs/1707.01495
    """
    
    def __init__(self, capacity: int, k_goals: int = 4,
                 strategy: str = 'future'):
        """
        Initialize HER buffer.
        
        Args:
            capacity: Maximum buffer size
            k_goals: Number of additional goals to sample per experience
            strategy: Goal sampling strategy ('future', 'final', or 'random')
        """
        super().__init__(capacity)
        self.k_goals = k_goals
        self.strategy = strategy
        self.current_episode = []
    
    def add(self, state: torch.Tensor, action: torch.Tensor,
            reward: float, next_state: torch.Tensor,
            done: bool, info: Optional[Dict] = None):
        """Add transition to current episode."""
        self.current_episode.append(
            Transition(state, action, reward, next_state, done, info)
        )
        
        if done:
            # Process episode with hindsight goals
            self._store_episode_with_hindsight()
            self.current_episode = []
    
    def _store_episode_with_hindsight(self):
        """Store episode transitions with additional hindsight goals."""
        episode_length = len(self.current_episode)
        
        # Store original transitions
        for t, transition in enumerate(self.current_episode):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity
        
        # Add hindsight goals
        for t, transition in enumerate(self.current_episode):
            # Sample k alternative goals
            future_indices = range(t + 1, episode_length) if self.strategy == 'future' else range(episode_length)
            goal_indices = np.random.choice(future_indices, size=self.k_goals, replace=True)
            
            for goal_idx in goal_indices:
                goal_state = self.current_episode[goal_idx].next_state
                # Compute new reward based on goal
                achieved = transition.next_state
                new_reward = self._compute_reward(achieved, goal_state)
                
                # Create new transition with hindsight goal
                new_transition = Transition(
                    self._append_goal(transition.state, goal_state),
                    transition.action,
                    new_reward,
                    self._append_goal(transition.next_state, goal_state),
                    transition.done,
                    {'goal': goal_state, **transition.info} if transition.info else {'goal': goal_state}
                )
                
                if len(self.buffer) < self.capacity:
                    self.buffer.append(None)
                self.buffer[self.position] = new_transition
                self.position = (self.position + 1) % self.capacity
    
    def _append_goal(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Append goal to state representation."""
        return torch.cat([state, goal], dim=-1)
    
    def _compute_reward(self, achieved: torch.Tensor, goal: torch.Tensor) -> float:
        """Compute reward based on achieved state and goal."""
        # Simple sparse reward
        return float(torch.all(torch.abs(achieved - goal) < 0.05))
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))
        
        return {
            'states': torch.stack(batch.state),
            'actions': torch.stack(batch.action),
            'rewards': torch.tensor(batch.reward),
            'next_states': torch.stack(batch.next_state),
            'dones': torch.tensor(batch.done),
            'infos': batch.info
        } 