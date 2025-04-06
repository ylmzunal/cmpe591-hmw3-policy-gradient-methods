import os
import numpy as np
import torch

# Performance optimizations for Apple Silicon
try:
    torch.backends.mps.enable_fallback_to_cpu = True
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
except:
    print("MPS backend not available, using CPU instead")

# Import original environment
from homework3 import Hw3Env
print("Using original environment from homework3.py")

from agent import Agent
from sac_agent import SacAgent

def test_algorithm(algorithm, model_path=None, num_episodes=10, render_mode="offscreen"):
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
    
    print(f"\n===== Testing {algorithm.upper()} model =====")
    
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
    
    print(f"\nResults for {algorithm.upper()}:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return avg_reward, success_rate

def test():
    """Test both VPG and SAC models"""
    # Testing parameters
    num_episodes = 10
    render_mode = "offscreen"  # Use "gui" to visualize
    
    # Test VPG
    vpg_reward, vpg_success_rate = test_algorithm("vpg", num_episodes=num_episodes, render_mode=render_mode)
    
    # Test SAC
    sac_reward, sac_success_rate = test_algorithm("sac", num_episodes=num_episodes, render_mode=render_mode)
    
    # Print comparison
    print("\n===== Comparison =====")
    print(f"VPG: Avg Reward = {vpg_reward:.2f}, Success Rate = {vpg_success_rate:.2f}")
    print(f"SAC: Avg Reward = {sac_reward:.2f}, Success Rate = {sac_success_rate:.2f}")
    
    # Determine which algorithm performed better
    if vpg_success_rate > sac_success_rate:
        print("VPG performed better in terms of success rate")
    elif sac_success_rate > vpg_success_rate:
        print("SAC performed better in terms of success rate")
    else:
        if vpg_reward > sac_reward:
            print("VPG performed better in terms of reward")
        elif sac_reward > vpg_reward:
            print("SAC performed better in terms of reward")
        else:
            print("Both algorithms performed equally")

if __name__ == "__main__":
    # Test both algorithms
    test() 