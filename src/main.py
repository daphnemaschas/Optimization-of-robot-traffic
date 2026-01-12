"""
Main entry point for the Multi-Agent RL training pipeline.
Orchestrates environment interaction and model optimization.
"""

import torch
import torch.optim as optim
import yaml
from pettingzoo.mpe import simple_spread_v3
from src.models import Actor, Critic
from src.utils.buffer import Memory
from src.utils.data_collector import collect_data
from src.utils.updater import update_model

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)["robot_traffic"]

def main():
    # Configuration
    N_AGENTS = config["n_agents"]
    OBS_DIM = config["obs_dim"]
    ACTION_DIM = config["action_dim"]
    GLOBAL_OBS_DIM = OBS_DIM * N_AGENTS
    
    LR = config["lr"]
    EPISODES = config["episodes"]
    STEPS_PER_EPISODE = config["steps_per_episodes"]
    
    # Initialization: MAPPO, agents usually share the same Actor/Critic networks
    actor = Actor(OBS_DIM, ACTION_DIM)
    critic = Critic(GLOBAL_OBS_DIM)
    
    actor_opt = optim.Adam(actor.parameters(), lr=LR)
    critic_opt = optim.Adam(critic.parameters(), lr=LR)
    
    memory = Memory()
    
    # Initialize Environment
    env = simple_spread_v3.env(continuous_actions=False, render_mode=None)
    
    print(f"Starting training on {simple_spread_v3.__name__}...")

    # Training Loop
    for episode in range(EPISODES):
        # Collect experience
        collect_data(env, actor, memory, num_episodes=1, max_steps=STEPS_PER_EPISODE)
        
        # Update models
        update_model(actor, critic, memory, actor_opt, 
                     critic_opt, config["gamma"], config["ppo_clip"],
                     config["update_epochs"])
        
        if episode % 10 == 0:
            print(f"Episode {episode} completed.")

    print("Training finished!")
    # Save models
    torch.save(actor.state_dict(), "actor_model.pth")

if __name__ == "__main__":
    main()