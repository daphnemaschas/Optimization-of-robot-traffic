"""
Module for reward processing and advantage estimation.
"""

import torch

def compute_returns(rewards, is_terminals, gamma=0.99, normalize=True):
    """
    Computes the discounted cumulative returns for each time step.
    Formula: G_t = r_t + gamma * G_{t+1}
    
    Args:
        rewards (list): List of rewards collected during the episode.
        is_terminals (list): List of boolean flags (True if the state was terminal).
        gamma (float): Discount factor.
        
    Returns:
        torch.Tensor: Normalized returns (targets for the critic).
    """
    returns = []
    discounted_reward = 0
    
    # Iterate backwards from the last step to the first
    for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
        if is_terminal:
            discounted_reward = 0
        
        discounted_reward = reward + (gamma * discounted_reward)
        returns.insert(0, discounted_reward)
        
    # Convert to tensor
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalization (Crucial for training stability)
    # This helps the critic converge faster by keeping targets in a small range
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
    return returns