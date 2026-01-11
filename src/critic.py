"""
Module for the Critic (global evaluation)

Only exits during the training part. Can see everything in order to judge whether the team situtation is great.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, total_obs_dim): # Sum the observations of all agents
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) # Return a grade (V)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)