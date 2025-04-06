import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.idx = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        # Convert single value tensors to scalars
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        if isinstance(done, torch.Tensor):
            done = done.item()
            
        # Convert tensors to numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        # Flatten if needed
        if state.ndim > 1:
            state = state.flatten()
        if action.ndim > 1:
            action = action.flatten()
        if next_state.ndim > 1:
            next_state = next_state.flatten()
            
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )
        
    def __len__(self):
        return self.size 