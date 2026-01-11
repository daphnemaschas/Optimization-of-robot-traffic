import pytest
import torch
import numpy as np
from src.utils.data_collector import get_global_state

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