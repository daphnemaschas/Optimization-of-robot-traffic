"""
Critic Module

The Critic evaluates the joint state of all agents to provide a baseline 
value (V-value) used for advantage estimation during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """
    Neural network that estimates the state-value function V(s).
    
    This model is only used during the training phase (Centralized Training).
    It processes the global observation (concatenation of all agents' views)
    to judge the overall quality of the team's situation.
    """
    def __init__(self, total_obs_dim): # Sum the observations of all agents
        """
        Args:
            total_obs_dim (int): Sum of observations from all agents (N_agents * obs_dim).
        """
        super().__init__()
        # First layer expands to a higher dimension to capture agent interactions
        self.fc1 = nn.Linear(total_obs_dim, 128)
        # Second layer compresses features
        self.fc2 = nn.Linear(128, 64)
        # Final layer outputs a single scalar
        self.fc3 = nn.Linear(64, 1) # Return a grade (V: State-value estimate)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Global state tensor of shape (batch_size, total_obs_dim).
        Returns:
            torch.Tensor: State-value estimate V(s) of shape (batch_size, 1).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No activation function on the last layer (Regression task)
        return self.fc3(x)