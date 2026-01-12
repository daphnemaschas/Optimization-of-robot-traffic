"""
Module: Model Update (Policy Optimization)

This module implements the learning logic using the MAPPO (Multi-Agent PPO) algorithm. 
The update process follows the Centralized Training principle:
- Critic Evaluation: The Centralized Critic analyzes the Global State to predict the expected return (State-Value V).
- Advantage Estimation: We compare this prediction with the actual rewards collected from the environment to compute the Advantage estimate.
- Policy Gradient: This advantage is used to update the Actor (Policy), reinforcing actions that performed better than the Critic's baseline.
"""
import torch
import torch.nn.functional as F
from src.utils.processing import compute_returns

def update_model(actor, critic, memory, actor_opt, critic_opt, gamma=0.99, eps_clip=0.2, epochs=4):
    """
    Performs a gradient descent step on both Actor and Critic networks.
    
    Args:
        actor (nn.Module): The policy network (decentralized).
        critic (nn.Module): The value network (centralized).
        memory (Memory): The buffer containing collected trajectories.
        gamma (float): Discount factor for future rewards.
    """
    # Convert memory lists to tensors for batch processing
    states, global_obs, actions, rewards, is_terminals = memory.get_tensors()
    
    # Compute Value Targets
    targets = compute_returns(rewards, is_terminals, gamma=gamma, normalize=True)
    
    # Compute reference values
    with torch.no_grad():
        old_probs = actor(states)
        old_log_probs = torch.distributions.Categorical(old_probs).log_prob(actions)
    
    # Optimization loop
    for _ in range(epochs):
        # Update Critic
        values = critic(global_obs).squeeze()
        critic_loss = F.mse_loss(values, targets)
        
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        
        # Update Actor
        advantages = targets - values.detach()
        
        new_probs = actor(states)
        new_log_probs = torch.distributions.Categorical(new_probs).log_prob(actions)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
    
    # Clear memory buffer after the update
    memory.clear()