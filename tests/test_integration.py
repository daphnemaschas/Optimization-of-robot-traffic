import pytest
import torch
import torch.optim as optim
from src.critic import Critic
from src.actor import Actor
from src.buffer import Memory
from src.utils.updater import update_model

def test_full_update_cycle():
    """
    Integration test: Ensures that Actor and Critic can be updated
    using data stored in Memory without dimension errors.
    """
    # 1. Setup dimensions
    obs_dim = 18
    action_dim = 5
    n_agents = 3
    global_obs_dim = obs_dim * n_agents
    
    # 2. Initialize components
    actor = Actor(obs_dim, action_dim)
    critic = Critic(global_obs_dim)
    memory = Memory()
    
    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    # 3. Mock data collection (simulating 10 steps)
    for _ in range(10):
        memory.store(
            state=torch.randn(obs_dim),
            global_state=torch.randn(global_obs_dim),
            action=1,
            reward=1.0,
            is_terminal=False
        )
    
    # 4. Execute update (This is the critical part)
    # If dimensions are wrong, it will crash here
    try:
        update_model(actor, critic, memory, actor_opt, critic_opt)
        success = True
    except Exception as e:
        print(f"Update failed with error: {e}")
        success = False
    
    assert success
    assert len(memory.states) == 0  # Memory should be cleared after update