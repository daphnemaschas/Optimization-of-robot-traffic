import pytest
import torch
import torch.optim as optim
from src.actor import Actor
from src.critic import Critic
from src.buffer import Memory
from src.utils.updater import update_model

def test_update_model_decreases_loss():
    """
    Test if the update_model function actually modifies weights 
    (indicating a gradient step was taken).
    """
    # Setup
    obs_dim = 18
    action_dim = 5
    n_agents = 3
    
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim * n_agents)
    memory = Memory()
    
    # Save initial weights to compare later
    initial_actor_weights = actor.fc1.weight.clone().detach()
    
    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    # Fill memory with dummy data (2 episodes of 5 steps)
    for _ in range(15):
        memory.store(
            state=torch.randn(obs_dim),
            global_state=torch.randn(obs_dim * n_agents),
            action=1,
            reward=1.0,
            is_terminal=False
        )
    
    # Execute update
    update_model(actor, critic, memory, actor_opt, critic_opt, epochs=2)
    
    # Verification 1: Weights have changed
    updated_actor_weights = actor.fc1.weight.detach()
    assert not torch.equal(initial_actor_weights, updated_actor_weights)
    
    # Verification 2: Memory has been cleared
    assert len(memory.states) == 0


def test_update_model_dimensions():
    """
    Checks if the internal math handles batches correctly.
    """
    obs_dim = 18
    action_dim = 5
    n_agents = 3
    batch_size = 64 # Big batch
    
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim * n_agents)
    memory = Memory()
    
    actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    
    # Simulate a batch
    for _ in range(batch_size):
        memory.store(
            state=torch.randn(obs_dim),
            global_state=torch.randn(obs_dim * n_agents),
            action=torch.randint(0, action_dim, (1,)).item(),
            reward=1.0,
            is_terminal=False
        )
    
    # Raise an exception if the calculus is mistaken
    try:
        update_model(actor, critic, memory, actor_opt, critic_opt)
        success = True
    except RuntimeError as e:
        pytest.fail(f"Dimension mismatch during update: {e}")
    
    assert success