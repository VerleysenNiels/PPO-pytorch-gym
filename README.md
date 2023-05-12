# Proximal Policy Optimization (PPO) for OpenAI Gym Environments using PyTorch

This repository contains an implementation of the Proximal Policy Optimization (PPO) algorithm for use in OpenAI Gym environments using PyTorch. The PPO algorithm is a reinforcement learning technique that has been shown to be effective in a wide range of tasks, including both continuous and discrete control problems.

## Overview

PPO is a policy gradient method that seeks to optimize a stochastic policy in an on-policy manner. The key idea behind PPO is to use a clipped surrogate objective function that ensures that the policy update does not deviate too far from the current policy, while still allowing for significant improvements in the policy.

The PPO algorithm involves two main components: a policy network and a value network. The policy network takes in the current state as input and outputs a probability distribution over the available actions. The value network takes in the current state as input and outputs an estimate of the expected cumulative reward from that state.

During training, the agent interacts with the environment and collects a set of trajectories. These trajectories are used to update the policy and value networks using the PPO algorithm. The PPO algorithm involves two main steps:

1. **Policy Evaluation**: The value network is used to estimate the expected cumulative reward for each state in the trajectory. These estimates are used to calculate the advantages for each action in the trajectory.

2. **Policy Improvement**: The policy network is updated using the clipped surrogate objective function, which encourages the policy to move towards actions that have higher advantages.

## Implementation Details

This implementation of the PPO algorithm uses the PyTorch library for neural network computations. The code is designed to be flexible and easy to use, allowing for customization of various hyperparameters and network architectures. 

There are two main files in this repository for using the PPO algorithm with different types of OpenAI Gym environments:

1. **`main.py`**: This file is used for generic OpenAI Gym environments for instance those that are in the Box2D category, these include classic control problems like the CartPole and Pendulum environments.

2. **`main_atari.py`**: This file is used for OpenAI Gym environments that are in the Atari category, these are classic video games like Breakout and Pong.

To use this implementation, run one of the main files with appropriate command line arguments (once I've added argument parsing :sweat_smile:). These command line arguments specify hyperparameters and other options for the PPO algorithm and the OpenAI Gym environment.


https://user-images.githubusercontent.com/26146888/225728444-7b762845-e32f-4dfb-814c-57a12459ed39.mp4

https://user-images.githubusercontent.com/26146888/237050171-ea751180-80f4-4b7f-af64-24bb08979b10.mp4


## References

For more information on the Proximal Policy Optimization algorithm, see the original paper by Schulman et al. (2017): ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347). 

For more information on using reinforcement learning in OpenAI Gym, see the official documentation: ["Using Reinforcement Learning (RL) in OpenAI Gym"](https://gym.openai.com/docs/).
