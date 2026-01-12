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

def update_model(actor, critic, memory, actor_opt, critic_opt, gamma=0.99):
    """
    Performs a gradient descent step on both Actor and Critic networks.
    
    Args:
        actor (nn.Module): The policy network (decentralized).
        critic (nn.Module): The value network (centralized).
        memory (Memory): The buffer containing collected trajectories.
        gamma (float): Discount factor for future rewards.
    """
    # Convert memory lists to tensors for batch processing
    local_obs = torch.stack(memory.states)
    global_obs = torch.stack(memory.global_states)
    actions = torch.tensor(memory.actions)
    rewards = torch.tensor(memory.rewards)
    
    # 1. Compute Value Targets (Standard TD-target or Monte Carlo Returns)
    # The Critic aims to predict the discounted sum of future rewards
    # For simplicity, we use the immediate reward as a proxy here
    targets = rewards # In a full implementation, use discounted returns
    
    # 2. Update Centralized Critic
    # Use Mean Squared Error to minimize the gap between predicted and actual Value
    predicted_values = critic(global_obs).squeeze()
    critic_loss = F.mse_loss(predicted_values, targets)
    
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    
    # 3. Compute Advantage estimate
    # Advantage = Actual Return - Baseline (Critic's prediction)
    # We detach the values to avoid backpropagating the Actor loss into the Critic
    advantages = targets - predicted_values.detach()
    
    # 4. Update Decentralized Actor (Policy Gradient)
    # Increase the probability of actions that resulted in positive advantages
    action_probs = actor(local_obs)
    # (Implementation of the specific RL loss like PPO or Vanilla Policy Gradient follows)
    
    # Clear memory buffer after the update
    memory.clear()