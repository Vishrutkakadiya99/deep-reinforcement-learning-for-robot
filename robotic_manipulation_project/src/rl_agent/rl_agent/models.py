# In file: src/rl_agent/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    The Actor network (Policy)
    Input: State vector (e.g., from CNN + robot joints)
    Output: Action (e.g., joint velocities)
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, action_dim)
        
        # Store max_action to scale the output
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Use torch.tanh to squash output between -1 and 1
        # Then scale it by the maximum allowed action value
        action = self.max_action * torch.tanh(self.fc_out(x))
        return action

class Critic(nn.Module):
    """
    The Critic network (Value function)
    Input: State vector and an Action
    Output: Q-Value (the expected future reward for that state-action pair)
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out1 = nn.Linear(256, 1)

    def forward(self, state, action):
        # Concatenate state and action along the last dimension
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1_value = self.fc_out1(q1)
        return q1_value
