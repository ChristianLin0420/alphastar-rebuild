import matplotlib.pyplot as plt
from torch.utils.tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from typing import List, Dict

class AlphaStarVisualizer:
    def __init__(self, log_dir: str):
        self.event_acc = EventAccumulator(log_dir)
        self.event_acc.Reload()
        
    def plot_training_curves(self, metrics: List[str], save_path: str = None):
        """Plot training curves for specified metrics."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            events = self.event_acc.Scalars(f'training/{metric}')
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            ax.plot(steps, values)
            ax.set_title(f'Training {metric}')
            ax.set_xlabel('Steps')
            ax.set_ylabel(metric)
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_evaluation_metrics(self, metrics: List[str], save_path: str = None):
        """Plot evaluation metrics over time."""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 5*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metrics):
            for stat in ['mean', 'min', 'max']:
                events = self.event_acc.Scalars(f'evaluation/{metric}/{stat}')
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                ax.plot(steps, values, label=stat)
            
            ax.set_title(f'Evaluation {metric}')
            ax.set_xlabel('Steps')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show() 