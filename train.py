import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Performance optimizations for Apple Silicon
# Enable MPS fallback - important for operations not supported natively by MPS
try:
    torch.backends.mps.enable_fallback_to_cpu = True
    # Set environment variables to optimize PyTorch performance
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Optimize memory allocation
    torch.mps.empty_cache()
except:
    print("MPS backend not available, using CPU instead")

# Set number of threads for intraop parallelism
torch.set_num_threads(8)  # Adjust based on your specific CPU

# Import original environment
from homework3 import Hw3Env
print("Using original environment from homework3.py")

from agent import Agent
from sac_agent import SacAgent

def train_vpg(num_episodes=1000, render_mode="offscreen", save_path="vpg_model.pt", fast_mode=True):
    """Train a Vanilla Policy Gradient (REINFORCE) agent"""
    env = Hw3Env(render_mode=render_mode, fast_mode=fast_mode)
    agent = Agent(lr=3e-4)
    
    # Explicitly set model to training mode
    agent.model.train()
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    all_rewards = []
    episode_rewards = []
    episode_times = []
    
    print(f"Starting VPG training for {num_episodes} episodes...")
    start_time = time.time()
    
    # Set save frequency - save every 1000 episodes for large training runs
    if num_episodes >= 10000:
        save_freq = 1000
    else:
        save_freq = max(1, min(100, num_episodes // 5))  # Save 5 times during training for smaller runs
    
    print(f"Models will be saved every {save_freq} episodes")
    
    for i in range(num_episodes):
        episode_start = time.time()
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        
        steps = 0  # Track steps per episode
        max_steps = 100 if fast_mode else 300  # Shorter episodes in fast mode
        
        while not done and steps < max_steps:
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            agent.add_reward(reward)
            cumulative_reward += reward
            done = is_terminal or is_truncated
            state = next_state
            steps += 1
            
            # In fast mode, break early if we hit max iter issues repeatedly
            if fast_mode and steps > 5 and reward < -1:
                break
        
        # Update policy after episode is complete
        loss = agent.update_model()
        
        # Log progress
        episode_end = time.time()
        episode_time = episode_end - episode_start
        episode_times.append(episode_time)
        episode_rewards.append(cumulative_reward)
        
        if (i+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_time = np.mean(episode_times[-10:])
            all_rewards.append(avg_reward)
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Episode Time: {avg_time:.2f}s, Steps: {steps}")
        
        # Save model periodically
        if (i+1) % save_freq == 0:
            checkpoint_path = f"models/vpg_checkpoint_{i+1}.pt"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    agent.save(f"models/{save_path}")
    print(f"Saved final model to models/{save_path}")
    
    # Save rewards for plotting
    np.save("models/vpg_rewards.npy", np.array(all_rewards))
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.xlabel('Episode (x10)')
    plt.ylabel('Average Reward (10 episodes)')
    plt.title('VPG Training Progress')
    plt.savefig("models/vpg_rewards.png")
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"\nVPG Training completed in {total_time:.2f} seconds")
    print(f"Average time per episode: {np.mean(episode_times):.2f} seconds")
    
    return agent, all_rewards

def train_sac(num_episodes=1000, render_mode="offscreen", save_path="sac_model.pt", fast_mode=True):
    """Train a Soft Actor-Critic agent"""
    env = Hw3Env(render_mode=render_mode, fast_mode=fast_mode)
    
    # Create SAC agent
    state_dim = 6  # Dimensionality of high_level_state
    action_dim = 2  # 2D continuous action space
    
    agent = SacAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=[256, 256],
        replay_buffer_size=100000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        alpha=0.2,
        automatic_entropy_tuning=True
    )
    
    # Explicitly set models to training mode
    agent.policy.train()
    agent.critic.train()
    
    # Create directory for models if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    all_rewards = []
    episode_rewards = []
    episode_times = []
    
    # Training loop
    updates = 0
    print(f"Starting SAC training for {num_episodes} episodes...")
    start_time = time.time()
    
    # Set save frequency - save every 1000 episodes for large training runs
    if num_episodes >= 10000:
        save_freq = 1000
    else:
        save_freq = max(1, min(100, num_episodes // 5))  # Save 5 times during training for smaller runs
    
    print(f"Models will be saved every {save_freq} episodes")
    
    for i in range(num_episodes):
        episode_start = time.time()
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        
        steps = 0  # Track steps per episode
        max_steps = 100 if fast_mode else 300  # Shorter episodes in fast mode
        
        while not done and steps < max_steps:
            # Select action
            action = agent.decide_action(state)
            
            # Take action
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            
            # Store transition
            done_bool = float(is_terminal or is_truncated)
            agent.add_experience(state, action, reward, next_state, done_bool)
            
            cumulative_reward += reward
            state = next_state
            done = is_terminal or is_truncated
            steps += 1
            
            # In fast mode, break early if we hit max iter issues repeatedly
            if fast_mode and steps > 5 and reward < -1:
                break
            
            # Update if enough samples
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update_model()
                updates += 1
        
        # Log progress
        episode_end = time.time()
        episode_time = episode_end - episode_start
        episode_times.append(episode_time)
        episode_rewards.append(cumulative_reward)
        
        if (i+1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_time = np.mean(episode_times[-10:])
            all_rewards.append(avg_reward)
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Episode Time: {avg_time:.2f}s, Steps: {steps}, Updates: {updates}")
        
        # Save model periodically
        if (i+1) % save_freq == 0:
            checkpoint_path = f"models/sac_checkpoint_{i+1}.pt"
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    agent.save(f"models/{save_path}")
    print(f"Saved final model to models/{save_path}")
    
    # Save rewards for plotting
    np.save("models/sac_rewards.npy", np.array(all_rewards))
    
    # Plot rewards
    plt.figure(figsize=(10, 5))
    plt.plot(all_rewards)
    plt.xlabel('Episode (x10)')
    plt.ylabel('Average Reward (10 episodes)')
    plt.title('SAC Training Progress')
    plt.savefig("models/sac_rewards.png")
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"\nSAC Training completed in {total_time:.2f} seconds")
    print(f"Average time per episode: {np.mean(episode_times):.2f} seconds")
    print(f"Total model updates: {updates}")
    
    return agent, all_rewards

def plot_comparison():
    """Compare VPG and SAC performance"""
    try:
        vpg_rewards = np.load("models/vpg_rewards.npy")
        sac_rewards = np.load("models/sac_rewards.npy")
        
        plt.figure(figsize=(12, 6))
        plt.plot(vpg_rewards, label='VPG (REINFORCE)')
        plt.plot(sac_rewards, label='SAC')
        plt.xlabel('Episode (x10)')
        plt.ylabel('Average Reward (10 episodes)')
        plt.title('VPG vs SAC Training Progress')
        plt.legend()
        plt.savefig("models/comparison.png")
        plt.show()
    except Exception as e:
        print(f"Couldn't load reward data for comparison: {str(e)}")

def train():
    """Main training function to train both algorithms in sequence"""
    # Training parameters - can be adjusted here
    num_episodes = 10000  # Full training run of 10000 episodes
    render_mode = "offscreen"
    fast_mode = True  # Enable fast mode by default
    
    print("===== Training VPG (REINFORCE) agent =====")
    train_vpg(num_episodes=num_episodes, render_mode=render_mode, fast_mode=fast_mode)
    
    print("\n===== Training SAC agent =====")
    train_sac(num_episodes=num_episodes, render_mode=render_mode, fast_mode=fast_mode)
    
    print("\n===== Generating comparison plot =====")
    plot_comparison()
    print("Training completed for both algorithms.")

if __name__ == "__main__":
    # Always train both algorithms
    train() 