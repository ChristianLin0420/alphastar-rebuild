import torch
import numpy as np
from typing import Dict, List
from ..env.sc2_env import SC2Environment

class AlphaStarEvaluator:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def evaluate_episode(self, env: SC2Environment) -> Dict[str, float]:
        """Evaluate model performance for one episode."""
        state = env.reset()
        done = False
        total_reward = 0
        metrics = {
            'win': 0,
            'army_value': 0,
            'resource_collection_rate': 0,
            'action_accuracy': []
        }
        
        while not done:
            with torch.no_grad():
                action_outputs = self.model(**state)
            
            next_state, reward, done, info = env.step(action_outputs)
            
            # Update metrics
            total_reward += reward
            metrics['army_value'] = max(metrics['army_value'], 
                                      state['scalar_input'][4].item())  # army_count
            metrics['resource_collection_rate'] += (
                state['scalar_input'][0].item() +  # minerals
                state['scalar_input'][1].item()    # vespene
            )
            
            state = next_state
        
        metrics['win'] = 1 if total_reward > 0 else 0
        metrics['resource_collection_rate'] /= env.env.steps
        metrics['total_reward'] = total_reward
        
        return metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate model over multiple episodes."""
        env = SC2Environment(self.config)
        all_metrics = []
        
        for _ in range(num_episodes):
            episode_metrics = self.evaluate_episode(env)
            all_metrics.append(episode_metrics)
        
        # Aggregate metrics
        aggregated_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        env.close()
        return aggregated_metrics 