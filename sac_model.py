import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=[256, 256]):
        super(QNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1_q1 = nn.Linear(obs_dim + action_dim, hidden_dim[0])
        self.linear2_q1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3_q1 = nn.Linear(hidden_dim[1], 1)
        
        # Q2 architecture (for double Q-learning)
        self.linear1_q2 = nn.Linear(obs_dim + action_dim, hidden_dim[0])
        self.linear2_q2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3_q2 = nn.Linear(hidden_dim[1], 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        x_action = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1_q1(x_action))
        x1 = F.relu(self.linear2_q1(x1))
        q1 = self.linear3_q1(x1)
        
        x2 = F.relu(self.linear1_q2(x_action))
        x2 = F.relu(self.linear2_q2(x2))
        q2 = self.linear3_q2(x2)
        
        return q1, q2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=[256, 256], action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(obs_dim, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        
        self.mean_linear = nn.Linear(hidden_dim[1], action_dim)
        self.log_std_linear = nn.Linear(hidden_dim[1], action_dim)
        
        self.apply(weights_init_)
        
        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Calculate log probability and entropy
        log_prob = normal.log_prob(x_t)
        
        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device) 