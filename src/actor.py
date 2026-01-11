"""
Actor Module

The Actor represents the agent's policy, mapping local observations 
to a probability distribution over possible actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Neural network representing the policy of an individual agent.
    
    This model follows the 'Decentralized Execution' principle: each agent 
    uses its own instance (or a shared one) to make decisions based solely 
    on its local observation.
    """
    def __init__(self, obs_dim, action_dim):
        """
        Args:
            obs_dim (int): Dimension of the agent's local observation vector.
            action_dim (int): Number of discrete actions available in the environment.
        """
        super().__init__()
        # Layers are sized to balance performance and computational cost
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # The output layer matches the number of possible actions
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        """
        Performs a forward pass to predict action probabilities.
        
        Args:
            x (torch.Tensor): Local observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            torch.Tensor: Normalized probability distribution over actions 
                          of shape (batch_size, action_dim).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Softmax to get a probability in the actions
        return F.softmax(self.fc3(x), dim=-1) # dim=-1 ensures Softmax is applied across the action dimension