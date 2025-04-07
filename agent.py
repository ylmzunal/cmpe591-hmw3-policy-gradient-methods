import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from model import VPG

gamma = 0.99

class Agent():
    def __init__(self, lr=3e-4):
        # Policy network and optimizer
        self.model = VPG()
        self.initial_lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        self.entropy_coef = 0.01  # Added entropy coefficient for better exploration
        self.update_count = 0  # Track the number of updates
        
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
            
        self.model.to(self.device)
        
    def decide_action(self, state):
        # Convert state to tensor if it's a numpy array
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # During evaluation, we don't need to track gradients
        with torch.set_grad_enabled(self.model.training):
            # Get mean and std from policy network
            action_mean, act_std = self.model(state).chunk(2, dim=-1)
            # Use softplus for std and increase minimum to encourage more exploration
            action_std = F.softplus(act_std) + 0.1  # Increased minimum std for better exploration
            
            # Create normal distribution
            dist = Normal(action_mean, action_std)
            
            # Sample action
            if self.model.training:
                action = dist.sample()
            else:
                # During evaluation, use the mean action for more stability
                action = action_mean
            
            # Clip action to valid range before calculating log_prob
            # This prevents extreme actions that might cause physical issues
            action_for_env = action.clamp(-1, 1)
            
            # Save log probability for training
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            if self.model.training:
                self.log_probs.append(log_prob)
                self.entropies.append(entropy)
        
        return action_for_env
    
    def _adjust_learning_rate(self):
        """Decrease learning rate based on the number of updates"""
        # Decay learning rate based on number of updates
        # This will reduce the learning rate to 10% of initial over 10000 episodes
        if self.update_count % 1000 == 0 and self.update_count > 0:
            # Calculate the new learning rate (exponential decay)
            progress = min(self.update_count / 10000, 1.0)  # Normalize to [0,1]
            new_lr = self.initial_lr * (0.1 ** progress)  # Decay to 10% of original
            
            # Update the optimizer with the new learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            print(f"Learning rate adjusted to {new_lr:.2e} after {self.update_count} updates")
    
    def update_model(self):
        # REINFORCE update
        if len(self.rewards) > 0:
            # Convert rewards to tensor
            R = 0
            returns = []
            
            # Calculate discounted rewards
            for r in self.rewards[::-1]:
                R = r + gamma * R
                returns.insert(0, R)
                
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Normalize returns for stable training
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # Calculate loss
            policy_loss = []
            entropy_loss = []
            for i, (log_prob, R, entropy) in enumerate(zip(self.log_probs, returns, self.entropies)):
                # Ensure log_prob requires grad
                if not log_prob.requires_grad:
                    # This generally shouldn't happen if your model is in training mode
                    # But we'll detach and create a new tensor with requires_grad=True
                    log_prob = log_prob.detach().requires_grad_(True)
                policy_loss.append(-log_prob * R)
                entropy_loss.append(-entropy)
                
            # Sum up all the losses - make sure we're working with tensors that require grad
            if len(policy_loss) > 0:
                # Combine policy loss with entropy loss for better exploration
                policy_loss = torch.stack(policy_loss).sum()
                entropy_loss = torch.stack(entropy_loss).sum() * self.entropy_coef
                loss = policy_loss + entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                
                # Track updates and adjust learning rate
                self.update_count += 1
                self._adjust_learning_rate()
                
                # Clear saved rewards and log_probs
                self.rewards = []
                self.log_probs = []
                self.entropies = []
                
                return loss.item()
            
        return 0.0

    def add_reward(self, reward):
        self.rewards.append(reward)
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        
