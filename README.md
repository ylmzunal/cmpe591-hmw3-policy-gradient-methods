# CMPE591 Homework 3: Policy Gradient Methods

This repository contains the implementation of policy gradient methods for a robotic manipulation task:
1. Vanilla Policy Gradient (REINFORCE)
2. Soft Actor-Critic (SAC)

## Task Description

The goal is to train a policy to push an object to a target position in a simulated environment. The environment is a robotic arm (UR5e) with a 2D continuous action space that controls the end-effector position. The state consists of the end-effector position, object position, and goal position (6-dimensional state vector).

## Implementation

### 1. Vanilla Policy Gradient (REINFORCE)

REINFORCE is a Monte Carlo policy gradient method that learns a policy by directly optimizing the expected returns. Key features:
- Gaussian policy network that outputs mean and standard deviation for actions
- Monte Carlo returns calculation with discount factor
- Return normalization for training stability

### 2. Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic method that incorporates entropy regularization. Key features:
- Twin Q-networks (critics) to mitigate overestimation bias
- Gaussian policy network with reparameterization trick
- Automatic entropy tuning
- Experience replay for improved sample efficiency

## Project Structure

- `agent.py`: REINFORCE implementation
- `model.py`: Policy network for REINFORCE
- `sac_agent.py`: SAC agent implementation
- `sac_model.py`: Actor and critic networks for SAC
- `replay_buffer.py`: Replay buffer for SAC
- `homework3.py`: Environment implementation
- `main.py`: Training and testing script

## How to Run

### Training

To train both algorithms:
```
python main.py --mode train --algo both
```

Train only VPG:
```
python main.py --mode train --algo vpg
```

Train only SAC:
```
python main.py --mode train --algo sac
```

### Testing

To test the trained models:
```
python main.py --mode test --algo vpg
python main.py --mode test --algo sac
```

To use GUI rendering during testing:
```
python main.py --mode test --algo vpg --render gui
```

### Plot Comparison

To compare the performance of both algorithms:
```
python main.py --mode plot
```

## Results

The training curves for both algorithms are saved in the `models` directory:
- `vpg_rewards.png`: VPG training curve
- `sac_rewards.png`: SAC training curve
- `comparison.png`: Comparison between VPG and SAC

### Performance Analysis

SAC generally outperforms VPG due to:
1. Off-policy learning which is more sample efficient
2. Experience replay to reuse past experiences
3. Twin critics to reduce overestimation bias
4. Entropy regularization to encourage exploration

However, VPG is simpler to implement and understand, and can perform well with proper hyperparameter tuning.

## Hyperparameters

### VPG
- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- Policy network: [256, 512, 256] hidden units

### SAC
- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- Soft update coefficient (tau): 0.005
- Batch size: 256
- Replay buffer size: 100,000
- Initial entropy coefficient (alpha): 0.2
- Hidden layers: [256, 256] 