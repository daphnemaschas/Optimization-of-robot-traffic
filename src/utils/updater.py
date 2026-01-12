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
    targets = compute_returns(rewards, is_terminals, gamma=gamma, normalize=False)
    
    # Compute reference values
    with torch.no_grad():
        old_values = critic(global_obs).squeeze()
        old_probs = actor(states)
        old_log_probs = torch.distributions.Categorical(old_probs).log_prob(actions)

        advantages = targets - old_values
        if advantages.size(0) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Optimization loop
    for _ in range(epochs):
        # Actor Update
        new_probs = actor(states)
        dist = torch.distributions.Categorical(new_probs)
        new_log_probs = dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

        # PPO Loss + Entropy Bonus (to prevent premature convergence to random)
        entropy = dist.entropy().mean()
        actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy # Reduced entropy here to prevent it from overpowering the learning on small samples
        
        actor_opt.zero_grad()
        actor_loss.backward()
        # Gradient clipping to prevent "explosions"
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_opt.step()

        # Critic Update
        current_values = critic(global_obs).squeeze()
        critic_loss = F.mse_loss(current_values, targets)
        
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_opt.step()

    # Clear memory buffer after the update
    memory.clear()