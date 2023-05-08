import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes the weights and biases of a given layer
    1. The weights with orthogonal initialization
    2. The biases are constant at 0
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentSmall(nn.Module):
    """
    PPO agent with a small neural network for both actor and critic.
    This is ideal for simple environments (with smaller state- and/or action-space)
    """
    def __init__(self, environments):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(environments.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(environments.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, environments.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        action_probabilities = Categorical(logits=logits)
        if action is None:
            action = action_probabilities.sample()
        return action, action_probabilities.log_prob(action), action_probabilities.entropy(), self.critic(x)
    
    def to(self, device):
        self.critic = self.critic.to(device)
        self.actor = self.actor.to(device)

class AtariAgentSmall(nn.Module):
    """
    PPO agent with a small convolutional neural network as single backbone for both actor and critic.
    Architecture is reused from original DQN paper
    """
    def __init__(self, environments):
        super().__init__()
        # Shared backbone --> performs feature extraction
        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU()
        )
        
        # Actor and critic heads
        self.actor = layer_init(nn.Linear(512, environments.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.backbone(x / 255.0))     # Scale observation to range of 0 to 1 by dividing by 255 

    def get_action_and_value(self, x, action=None):
        hidden = self.backbone(x / 255.0)    # Scale observation to range of 0 to 1 by dividing by 255 
        logits = self.actor(hidden)
        action_probabilities = Categorical(logits=logits)
        if action is None:
            action = action_probabilities.sample()
        return action, action_probabilities.log_prob(action), action_probabilities.entropy(), self.critic(hidden)
    
    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.critic = self.critic.to(device)
        self.actor = self.actor.to(device)