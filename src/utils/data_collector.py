"""
Utility module for Data Collection.

Handles the interaction between agents and the environment. 
To comply with CTDE, each agent's observation is collected locally for the Actor,
while joint observations are grouped to form the global state for the Critic.
"""

import torch
import numpy as np
from ..buffer import Memory
from pettingzoo.mpe import simple_spread_v3

env = simple_spread_v3.env(render_mode="rgb_array", continuous_actions=False)
memory = Memory()

def get_global_state(obs_dict):
    """
    Concatenates observations from all agents into a single flat vector.
    
    Args:
        obs_dict (dict): Dictionary mapping agent names to their local observations.
        
    Returns:
        torch.Tensor: A flattened tensor representing the joint state of the team.
    """
    sorted_keys = sorted(obs_dict.keys())
    global_state = np.concatenate([obs_dict[agent] for agent in sorted_keys])
    return torch.FloatTensor(global_state)


def collect_data(env, actor, memory, num_episodes=1, max_steps=25):
    """
    Runs episodes and stores transitions in the provided memory buffer.
    
    Args:
        env: The PettingZoo environment instance.
        actor (nn.Module): The policy network used to select actions.
        memory (Memory): The buffer object where transitions are stored.
        num_episodes (int): Number of complete episodes to run.
        max_steps (int): Maximum number of environment steps per episode.
    """
    for episode in range(num_episodes):
        env.reset()
        
        # Initialize the global state tracker with zeros
        current_obs = {agent: np.zeros(env.observation_space(agent).shape) for agent in env.agents}
        
        agent_step_count = {agent: 0 for agent in env.agents}
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            current_obs[agent] = obs
            
            if termination or truncation:
                env.step(None)
                continue # Go to next agent
            
            if agent_step_count[agent] >= max_steps:
                # To stop PettingZoo properly without sending None to an active agent,
                # we just need to break the agent_iter loop.
                break

            else:
                # 1. The agent looks at their own observation (Local View)
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0) # Add batch dimension
                
                # 2. The Actor chooses an action (here simulating with random for now)
                with torch.no_grad():
                    action_probs = actor(obs_tensor)
                
                # Sampling introduces exploration (stochastic policy)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                
                # 3. Store for future training
                global_state = get_global_state(current_obs)
                memory.store(
                    state=torch.FloatTensor(obs),
                    global_state=global_state,
                    action=action,
                    reward=reward,
                    is_terminal=termination
                )

            env.step(action)
            agent_step_count[agent] += 1