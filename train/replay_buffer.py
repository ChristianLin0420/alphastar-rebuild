import numpy as np
from collections import deque
import random
from typing import Dict, List

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: Dict, action: Dict, reward: float, 
             next_state: Dict, done: bool):
        """Save a transition."""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List:
        """Sample a batch of transitions."""
        transitions = random.sample(self.buffer, batch_size)
        batch = list(zip(*transitions))
        
        # Convert to appropriate format
        states = self._stack_dicts(batch[0])
        actions = self._stack_dicts(batch[1])
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = self._stack_dicts(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def _stack_dicts(self, dicts: List[Dict]) -> Dict:
        """Stack dictionary values into batches."""
        result = {}
        for key in dicts[0].keys():
            if isinstance(dicts[0][key], torch.Tensor):
                result[key] = torch.stack([d[key] for d in dicts])
            elif isinstance(dicts[0][key], tuple):
                result[key] = tuple(torch.stack([d[key][i] for d in dicts])
                                  for i in range(len(dicts[0][key])))
        return result
    
    def __len__(self) -> int:
        return len(self.buffer) 