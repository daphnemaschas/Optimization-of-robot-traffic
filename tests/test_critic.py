import torch
import pytest
from src.critic import Critic

def test_critic_output_dimension():
    """
    Test if the Centralized Critic processes the global state 
    and returns a single scalar value.
    """
    num_agents = 3
    obs_dim = 18
    global_obs_dim = num_agents * obs_dim # 54
    
    critic = Critic(global_obs_dim)
    
    # Create a dummy global state (batch size of 1)
    dummy_global_obs = torch.randn(1, global_obs_dim)
    
    value = critic(dummy_global_obs)
    
    # The output must be (batch_size, 1)
    assert value.shape == (1, 1)
    assert isinstance(value, torch.Tensor)

def test_critic_forward_batch():
    """
    Test if the Critic can handle a batch of global states (e.g., during training).
    """
    global_obs_dim = 54
    batch_size = 32
    critic = Critic(global_obs_dim)
    
    batch_global_obs = torch.randn(batch_size, global_obs_dim)
    values = critic(batch_global_obs)
    
    assert values.shape == (batch_size, 1)