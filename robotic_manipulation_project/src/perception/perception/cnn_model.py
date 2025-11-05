# In file: src/perception/cnn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptionCNN(nn.Module):
    """
    A simple CNN to extract features from an 84x84 image.
    Input: (Batch, 3, 84, 84) - A batch of 84x84 RGB images
    Output: (Batch, 256) - A compact state vector
    """
    def __init__(self, input_channels=3, output_dim=256):
        super(PerceptionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4) # -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)              # -> (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)              # -> (64, 7, 7)
        
        # Calculate the flattened size after conv layers
        self.flattened_size = 64 * 7 * 7
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc_out = nn.Linear(512, output_dim)

    def forward(self, x):
        # Input images are usually (0, 255), so normalize to (0, 1)
        x = x / 255.0
        
        # Apply conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully connected layers
        # .view(-1, ...) keeps the batch dimension and flattens the rest
        x = x.view(-1, self.flattened_size)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        state_vector = self.fc_out(x)
        
        return state_vector
