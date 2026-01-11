import pytest
import torch
from src.utils.processing import compute_returns

def test_compute_returns():
    rewards = [1.0, 1.0, 1.0]
    is_terminals = [False, False, True]
    gamma = 0.9
    # Expected: 
    # R2 = 1
    # R1 = 1 + 0.9*1 = 1.9
    # R0 = 1 + 0.9*1.9 = 2.71
    expected = torch.tensor([2.71, 1.9, 1.0])
    
    returns = compute_returns(rewards,is_terminals, gamma, normalize=False)
    assert torch.allclose(returns, expected)