import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sac_model import GaussianPolicy, QNetwork
from replay_buffer import ReplayBuffer

class SacAgent:
    def __init__(
        self,
        state_dim=6,
        action_dim=2,
        hidden_dim=[256, 256],
        replay_buffer_size=1000000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=True
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Use Apple MPS if available, else use CUDA if available, else use CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple GPU) for training")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA for training")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for training")
        
        # Initialize critics
        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize critic target with the same weights
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Initialize actor
        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Initialize optimizers with slightly lower learning rate for stability
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr*0.5)  # Lower lr for policy
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor([action_dim], device=self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_dim, action_dim, self.device)
        
        # For training statistics
        self.rewards = []
        self.critic_losses = []
        self.policy_losses = []
        self.alpha_losses = []
        
    def decide_action(self, state, evaluate=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            if evaluate:
                _, _, action = self.policy.sample(state)
            else:
                action, _, _ = self.policy.sample(state)
            
            # Ensure actions are within valid range to avoid environment issues
            action = action.clamp(-1, 1)
        
        return action
    
    def update_model(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
            
        # Update critics
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state)
            
            # Target Q-values
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Calculate critic loss - scale down to prevent exploding gradients
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor less frequently for stability
        if np.random.rand() < 0.5:  # Only update policy 50% of the time
            # Update actor
            pi, log_pi, _ = self.policy.sample(state)
            q1_pi, q2_pi = self.critic(state, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            
            # Calculate policy loss
            policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
            
            # Update actor
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update alpha if automatic entropy tuning
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
                alpha_loss_value = alpha_loss.item()
            else:
                alpha_loss_value = 0
                
            # Store policy losses for statistics
            self.policy_losses.append(policy_loss.item())
            self.alpha_losses.append(alpha_loss_value)
        else:
            policy_loss = torch.tensor(0.0)
            alpha_loss_value = 0
            
        # Update target networks with smaller tau for more stable learning
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau*0.5) + param.data * self.tau*0.5)
            
        # Store losses for statistics
        self.critic_losses.append(critic_loss.item())
        
        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss,
            'alpha_loss': alpha_loss_value
        }
    
    def add_experience(self, state, action, reward, next_state, done):
        # Clip reward for stability
        reward = np.clip(reward, -10, 10)
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.rewards.append(reward)
        
    def add_reward(self, reward):
        self.rewards.append(reward)
        
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict']) 