import pytest
import torch
import torch.optim as optim
import numpy as np
from src.actor import Actor
from src.critic import Critic
from src.buffer import Memory
from src.utils.updater import update_model

def test_actor_learning_scenario():
    """
    Scenario: 'The Reward Seeker'
    If an agent is consistently rewarded for choosing action 1,
    its probability of choosing action 1 must increase.
    """
    # Setup
    obs_dim, action_dim, n_agents = 18, 5, 3
    actor = Actor(obs_dim, action_dim)
    critic = Critic(obs_dim * n_agents)
    memory = Memory()
    
    actor_opt = optim.Adam(actor.parameters(), lr=5e-3)
    critic_opt = optim.Adam(critic.parameters(), lr=5e-3)
    
    # Observe initial probability for action 1
    test_obs = torch.randn(obs_dim)
    with torch.no_grad():
        initial_probs = actor(test_obs.unsqueeze(0)).squeeze()
        prob_before = initial_probs[1].item()

    # Simulation: 20 steps where action 1 is ALWAYS the winner
    for _ in range(20):
        memory.store(
            state=test_obs.squeeze(),
            global_state=torch.randn(obs_dim * n_agents),
            action=1, # The rewarded action
            reward=20.0, # High reward
            is_terminal=False
        )

    # Update logic
    for _ in range(5):
        update_model(actor, critic, memory, actor_opt, critic_opt, epochs=10)
        # Refill memory for the next loop to keep the signal strong
        for _ in range(20):
            memory.store(test_obs, torch.randn(obs_dim * n_agents), 1, 20.0, False)

    # Check result
    with torch.no_grad():
        final_probs = actor(test_obs.unsqueeze(0)).squeeze()
        prob_after = final_probs[1].item()

    assert prob_after > prob_before, f"Probability decreased: {prob_before} -> {prob_after}"

def test_critic_value_convergence():
    """
    Scenario: 'The Oracle'
    If an agent always receives a return of 100, the Critic 
    should eventually predict a value close to 100.
    """
    obs_dim, n_agents = 18, 3
    critic = Critic(obs_dim * n_agents)
    optimizer = optim.Adam(critic.parameters(), lr=1e-2)
    
    global_obs = torch.randn(1, obs_dim * n_agents)
    target_value = torch.tensor([100.0])

    # Supervised learning loop (internal to Critic)
    for _ in range(100):
        predicted = critic(global_obs).squeeze()
        loss = torch.nn.functional.mse_loss(predicted, target_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_prediction = critic(global_obs).item()
    assert abs(final_prediction - 100.0) < 1.0