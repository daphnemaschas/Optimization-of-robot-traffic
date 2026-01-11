"""
Utilitary module for Data Collection

The agents interact with the environment. To comply with the CTDE, each agent's observation is collected,
and also grouped together for the critic.
"""

import torch
import numpy as np
from ..buffer import Memory
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="rgb_array", continuous_actions=False)
memory = Memory()

def get_global_state(obs_dict):
    """
    Takes the dictionary of observations from all agents 
    and transforms them into a single flat vector.
    """
    sorted_keys = sorted(obs_dict.keys())
    global_state = np.concatenate([obs_dict[agent] for agent in sorted_keys])
    
    return torch.FloatTensor(global_state)

def collect_data(num_episodes=10):
    for episode in range(num_episodes):
        env.reset()
        
        # All agents' observations are stored to create the overall status.
        current_obs = {agent: np.zeros(env.observation_space(agent).shape) for agent in env.agents}
        
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            current_obs[agent] = obs
            
            if termination or truncation:
                action = None
            else:
                # 1. The agent looks at their own observation (Local View)
                obs_tensor = torch.FloatTensor(obs)
                
                # 2. The Actor chooses an action (here simulating with random for now)
                action = env.action_space(agent).sample() 
                
                # 3. Store for future training
                global_state = get_global_state(current_obs)
                if memory is not None:
                    memory.states.append(obs_tensor)
                    memory.global_states.append(global_state)
                    memory.actions.append(action)
                    memory.rewards.append(reward)
                    memory.is_terminals.append(termination)

            env.step(action)