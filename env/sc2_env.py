from pysc2.env import sc2_env
from pysc2.lib import actions, features
import numpy as np
from typing import Dict, Tuple
from ..data.preprocessor import StarCraftPreprocessor

class SC2Environment:
    def __init__(self, config):
        self.config = config
        self.preprocessor = StarCraftPreprocessor(config)
        
        # Initialize SC2 environment
        self.env = sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=config.SPATIAL_SIZE,
                                                    minimap=config.SPATIAL_SIZE),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True
        )
        
    def reset(self) -> Dict:
        """Reset environment and return initial observation."""
        obs = self.env.reset()[0]
        return self._process_observation(obs)
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """Execute action and return new observation, reward, done flag, and info."""
        # Convert model output to SC2 action
        sc2_action = self._convert_action(action)
        
        # Execute action
        obs = self.env.step([sc2_action])[0]
        
        # Process results
        processed_obs = self._process_observation(obs)
        reward = float(obs.reward)
        done = obs.step_type == sc2_env.StepType.LAST
        info = {}
        
        return processed_obs, reward, done, info
    
    def _process_observation(self, obs) -> Dict:
        """Convert SC2 observation to model input format."""
        return {
            'scalar_input': self.preprocessor.preprocess_scalar_features({
                'minerals': obs.observation.player.minerals,
                'vespene': obs.observation.player.vespene,
                'food_used': obs.observation.player.food_used,
                'food_cap': obs.observation.player.food_cap,
                'army_count': obs.observation.player.army_count,
                'worker_count': obs.observation.player.worker_count
            }),
            'entity_input': self.preprocessor.preprocess_entity_features(
                obs.observation.feature_units
            ),
            'spatial_input': self.preprocessor.preprocess_spatial_features(
                obs.observation.feature_minimap
            )
        }
    
    def _convert_action(self, model_action: Dict) -> actions.FunctionCall:
        """Convert model output to SC2 action format."""
        action_type = model_action['action_type'].argmax().item()
        
        # Get action arguments based on action type
        args = []
        if actions.FUNCTIONS[action_type].args:
            if 'screen' in str(actions.FUNCTIONS[action_type].args):
                # Convert target_location to screen coordinates
                target = model_action['target_location'].view(-1, 2)
                args.append([int(target[0, 0]), int(target[0, 1])])
            
            if 'select_unit_id' in str(actions.FUNCTIONS[action_type].args):
                # Convert selected_units to unit IDs
                selected = model_action['selected_units'].argmax().item()
                args.append([selected])
        
        return actions.FunctionCall(action_type, args)
    
    def close(self):
        """Close environment."""
        self.env.close() 