"""
Memory buffer for storing transitions collected during episodes.
"""
import torch

class Memory:
    def __init__(self):
        # Storage for transitions
        self.states = []        # Local observations (Actor)
        self.global_states = [] # Global observations (Critic)
        self.actions = []
        self.rewards = []
        self.is_terminals = []

    def store(self, state, global_state, action, reward, is_terminal):
        """
        Stores a single step transition.
        """
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        """
        Clears the buffer. To be called after each update_model.
        """
        self.states.clear()
        self.global_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        
    def get_tensors(self):
        """
        Converts all stored lists into PyTorch Tensors for the Updater.
        """
        return (
            torch.stack(self.states).detach(),
            torch.stack(self.global_states).detach(),
            torch.tensor(self.actions).detach(),
            torch.tensor(self.rewards).detach(),
            torch.tensor(self.is_terminals).detach()
        )