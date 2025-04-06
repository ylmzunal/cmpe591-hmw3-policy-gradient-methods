import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Performance optimizations for Apple Silicon
# Enable MPS fallback - important for operations not supported natively by MPS
torch.backends.mps.enable_fallback_to_cpu = True

# Set environment variables to optimize PyTorch performance
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Set number of threads for intraop parallelism (M3 has 8 or 10 cores depending on model)
torch.set_num_threads(8)  # Adjust based on your specific M3 chip

# Optimize memory allocation
torch.mps.empty_cache()

from homework3 import Hw3Env
from agent import Agent
from sac_agent import SacAgent

def train_vpg(num_episodes=200, render_mode="offscreen", save_path="vpg_model.pt"):
    """Train a Vanilla Policy Gradient (REINFORCE) agent"""
    env = Hw3Env(render_mode=render_mode)
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
    
    for i in range(num_episodes):
        episode_start = time.time()
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        
        while not done:
            action = agent.decide_action(state)
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            agent.add_reward(reward)
            cumulative_reward += reward
            done = is_terminal or is_truncated
            state = next_state
        
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
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Episode Time: {avg_time:.2f}s")
        
        # Save model periodically
        if (i+1) % 100 == 0:
            agent.save(f"models/vpg_checkpoint_{i+1}.pt")
    
    # Save final model
    agent.save(f"models/{save_path}")
    
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

def train_sac(num_episodes=200, render_mode="offscreen", save_path="sac_model.pt"):
    """Train a Soft Actor-Critic agent"""
    env = Hw3Env(render_mode=render_mode)
    
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
    
    for i in range(num_episodes):
        episode_start = time.time()
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        
        while not done:
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
            print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Episode Time: {avg_time:.2f}s, Updates: {updates}")
        
        # Save model periodically
        if (i+1) % 100 == 0:
            agent.save(f"models/sac_checkpoint_{i+1}.pt")
    
    # Save final model
    agent.save(f"models/{save_path}")
    
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

def test(algorithm="vpg", model_path=None, num_episodes=10, render_mode="offscreen"):
    """Test a trained model"""
    env = Hw3Env(render_mode=render_mode)
    
    if algorithm.lower() == "vpg":
        if model_path is None:
            model_path = "models/vpg_model.pt"
        agent = Agent()
        agent.load(model_path)
        # Set to evaluation mode
        agent.model.eval()
    elif algorithm.lower() == "sac":
        if model_path is None:
            model_path = "models/sac_model.pt"
        agent = SacAgent()
        agent.load(model_path)
        # Set to evaluation mode
        agent.policy.eval()
        agent.critic.eval()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    episode_rewards = []
    success_count = 0
    
    for i in range(num_episodes):
        env.reset()
        state = env.high_level_state()
        done = False
        cumulative_reward = 0.0
        steps = 0
        
        while not done:
            # Select action
            if algorithm.lower() == "vpg":
                action = agent.decide_action(state)
            else:  # SAC
                action = agent.decide_action(state, evaluate=True)
            
            # Take action
            next_state, reward, is_terminal, is_truncated = env.step(action[0])
            
            cumulative_reward += reward
            state = next_state
            done = is_terminal or is_truncated
            steps += 1
            
            # Check for success
            if is_terminal and not is_truncated:
                success_count += 1
        
        episode_rewards.append(cumulative_reward)
        print(f"Episode {i+1}: Reward = {cumulative_reward:.2f}, Steps = {steps}")
    
    avg_reward = np.mean(episode_rewards)
    success_rate = success_count / num_episodes
    
    print(f"\nTesting Results for {algorithm.upper()}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return avg_reward, success_rate

def train(algorithm="both"):
    """Main training function with choice of algorithm"""
    if algorithm.lower() == "vpg" or algorithm.lower() == "both":
        print("Training VPG (REINFORCE) agent...")
        train_vpg()
    
    if algorithm.lower() == "sac" or algorithm.lower() == "both":
        print("Training SAC agent...")
        train_sac()

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
    except:
        print("Couldn't load reward data for comparison.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test RL agents for CMPE591 HW3')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'plot'],
                        help='Mode: train, test, or plot comparison')
    parser.add_argument('--algo', type=str, default='both', choices=['vpg', 'sac', 'both'],
                        help='Algorithm to use: vpg, sac, or both')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model for testing')
    parser.add_argument('--episodes', type=int, default=2000,
                        help='Number of episodes for training or testing')
    parser.add_argument('--render', type=str, default="offscreen", choices=["offscreen", "gui"],
                        help='Render mode: offscreen or gui')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                        help='Use GPU (MPS for Apple Silicon) if available')
    parser.add_argument('--cpu-only', action='store_true', 
                        help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Set device preferences based on arguments
    if args.cpu_only:
        # Override device selection in agents
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        print("Forcing CPU usage as per command line argument")
    elif not args.use_gpu:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
        print("GPU usage disabled via command line argument")
    
    if args.mode == 'train':
        train(args.algo)
    elif args.mode == 'test':
        test(args.algo, args.model, args.episodes, args.render)
    elif args.mode == 'plot':
        plot_comparison() 