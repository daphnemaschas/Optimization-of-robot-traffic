import pytest
import torch
import numpy as np
from src.utils.data_collector import get_global_state, collect_data
from src.buffer import Memory
from src.actor import Actor
import pettingzoo.mpe.simple_spread_v3 as simple_spread_v3

def test_get_global_state_shape():
    """
    Test if the global state concatenation returns the correct dimensions.
    """
    # Mock observations for 3 agents with 5 features each
    obs_dict = {
        "agent_0": np.array([1, 1, 1, 1, 1]),
        "agent_1": np.array([2, 2, 2, 2, 2]),
        "agent_2": np.array([3, 3, 3, 3, 3])
    }
    
    expected_dim = 15 # 3 agents * 5 features
    global_state = get_global_state(obs_dict)
    
    assert isinstance(global_state, torch.Tensor)
    assert global_state.shape[0] == expected_dim
    # Check if ordering is preserved (agent_0, then agent_1, then agent_2)
    assert global_state[0] == 1
    assert global_state[10] == 3


def test_collect_data_integration():
    """
    Test the full collection loop: env <-> actor <-> memory.
    """
    # 1. Setup
    env = simple_spread_v3.env(continuous_actions=False)
    obs_dim = 18
    action_dim = 5
    actor = Actor(obs_dim, action_dim)
    memory = Memory()
    
    max_steps = 5
    num_agents = 3
    
    # 2. Run collection
    collect_data(env, actor, memory, num_episodes=1, max_steps=max_steps)
    
    # 3. Verifications
    # Total steps stored should be: max_steps * num_agents
    expected_size = max_steps * num_agents
    
    assert len(memory.states) == expected_size
    assert len(memory.global_states) == expected_size
    assert len(memory.actions) == expected_size
    
    # Check if stored items are Tensors
    assert isinstance(memory.states[0], torch.Tensor)
    assert isinstance(memory.global_states[0], torch.Tensor)
    
    # Check dimensions
    assert memory.states[0].shape == (obs_dim,)
    assert memory.global_states[0].shape == (obs_dim * num_agents,)


def test_collect_data_empty_on_termination():
    """
    Ensure the loop handles env termination correctly.
    """
    # Simulate an environnement which terminates immediatly
    env = simple_spread_v3.env(continuous_actions=False)
    actor = Actor(18, 5)
    memory = Memory()
    
    # We ensure an immediate termination with a small number of steps
    collect_data(env, actor, memory, num_episodes=1, max_steps=0)
    
    assert len(memory.states) == 0