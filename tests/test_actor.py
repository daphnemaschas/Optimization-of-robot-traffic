import torch
import pytest
from src.actor import Actor

def test_actor_output_logic():
    """
    Test if the Actor network produces a valid probability distribution.
    """
    obs_dim = 18
    action_dim = 5
    actor = Actor(obs_dim, action_dim)
    
    # Create a dummy local observation (batch size of 1)
    dummy_obs = torch.randn(1, obs_dim)
    
    probs = actor(dummy_obs)
    
    # Test shape: (batch_size, action_dim)
    assert probs.shape == (1, action_dim)
    
    # Test Softmax constraint: Sum of probabilities must be 1
    assert torch.allclose(probs.sum(), torch.tensor(1.0)) # allclose because of floating point precision
    
    # Test Positivity: All probabilities must be >= 0
    assert torch.all(probs >= 0)

def test_actor_batch_processing():
    """
    Test if the Actor can process a batch of observations.
    """
    obs_dim = 18
    action_dim = 5
    batch_size = 32
    actor = Actor(obs_dim, action_dim)
    
    batch_obs = torch.randn(batch_size, obs_dim)
    probs = actor(batch_obs)
    
    assert probs.shape == (batch_size, action_dim)
    # Check if each row in the batch sums to 1
    row_sums = probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(batch_size))