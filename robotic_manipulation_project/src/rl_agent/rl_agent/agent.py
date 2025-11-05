# In file: src/rl_agent/agent.py

import torch
import torch.optim as optim
import numpy as np
from rl_agent.models import Actor, Critic # Import models from this package

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DRLAgent:
    """
    A simplified DRL Agent (like DDPG or SAC).
    This template includes action selection and placeholders for training.
    """
    def __init__(self, state_dim, action_dim, max_action):
        
        # Initialize Actor and Critic Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        
        # --- In a real DRL agent, you would also initialize: ---
        # 1. Target networks for stable learning
        # self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        # self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # 2. Optimizers for each network
        # self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        # self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 3. A Replay Buffer to store experiences
        # self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        
        print("DRL Agent Initialized.")
        print(f"Using device: {device}")

    def select_action(self, state):
        """
        Select an action based on the current state.
        state: A numpy array
        """
        # Convert state to a PyTorch tensor and send to device
        # Add a batch dimension (B=1)
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        # Get action from actor network (in no_grad mode for inference)
        with torch.no_grad():
            action = self.actor(state_tensor)
            
        # Convert action to a numpy array to send to ROS
        return action.cpu().data.numpy().flatten()

    def add_experience_to_buffer(self, state, action, next_state, reward, done):
        """
        Placeholder for adding an experience to the replay buffer.
        """
        # In a real implementation:
        # self.replay_buffer.add(state, action, next_state, reward, done)
        pass

    def train(self):
        """
        Placeholder for the training (update) step.
        """
        # In a real implementation (e.g., SAC/DDPG):
        # 1. Sample a batch from the replay_buffer
        #    (state, action, next_state, reward, done) = self.replay_buffer.sample(batch_size)
        # 2. Compute target Q-values using the target networks
        # 3. Compute current Q-values using the critic
        # 4. Compute critic loss (e.g., MSE) and update critic
        # 5. Compute actor loss (e.g., by maximizing Q-value) and update actor
        # 6. Soft-update target networks
        pass
        
    def save(self, filename):
        """Saves the actor and critic models."""
        torch.save(self.actor.state_dict(), f"{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filename}_critic.pth")
        print(f"Models saved to {filename}_actor/critic.pth")

    def load(self, filename):
        """Loads the actor and critic models."""
        self.actor.load_state_dict(torch.load(f"{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{filename}_critic.pth"))
        print(f"Models loaded from {filename}_actor/critic.pth")
