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

## Usage
Classic control:
```
usage: main.py [-h] [--environment-id ENVIRONMENT_ID] [--num-envs NUM_ENVS] [--num-steps-collected NUM_STEPS_COLLECTED] [--num-mini-batches NUM_MINI_BATCHES] [--num-epochs NUM_EPOCHS] [--learning-rate LEARNING_RATE] [--total-timesteps TOTAL_TIMESTEPS]
               [--gae-gamma GAE_GAMMA] [--gae-lambda GAE_LAMBDA] [--clip-coeff CLIP_COEFF] [--ent-loss-coeff ENT_LOSS_COEFF] [--value-loss-coeff VALUE_LOSS_COEFF] [--max-gradient MAX_GRADIENT] [--seed SEED] [--deterministic-torch] [--use-gpu]

PPO Classic Control Training Script

options:
  -h, --help            show this help message and exit
  --environment-id ENVIRONMENT_ID
                        Gym environment id to train on (default: CartPole-v1)
  --num-envs NUM_ENVS   number of parallel environments to use (default: 4)
  --num-steps-collected NUM_STEPS_COLLECTED
                        number of steps to collect per environment (default: 128)
  --num-mini-batches NUM_MINI_BATCHES
                        number of mini-batches to split each rollout into for training (default: 4)
  --num-epochs NUM_EPOCHS
                        number of epochs to train on each mini-batch (default: 4)
  --learning-rate LEARNING_RATE
                        learning rate (default: 2.5e-4)
  --total-timesteps TOTAL_TIMESTEPS
                        total number of timesteps to train for (default: 25000)
  --gae-gamma GAE_GAMMA
                        discount factor for Generalized Advantage Estimation (default: 0.99)
  --gae-lambda GAE_LAMBDA
                        Lambda parameter for Generalized Advantage Estimation (default: 0.95)
  --clip-coeff CLIP_COEFF
                        Clipping parameter for PPO loss function (default: 0.2)
  --ent-loss-coeff ENT_LOSS_COEFF
                        Weighting parameter for entropy loss term in PPO loss function (default: 0.01)
  --value-loss-coeff VALUE_LOSS_COEFF
                        Weighting parameter for value loss term in PPO loss function (default: 0.5)
  --max-gradient MAX_GRADIENT
                        Maximum gradient norm for gradient clipping (default: 0.5)
  --seed SEED           random seed (default: 1)
  --deterministic-torch
                        whether to use deterministic torch operations (default: False)
  --use-gpu             whether to use a GPU (default: False)
```

Atari:
```
usage: main_atari.py [-h] [--environment-id ENVIRONMENT_ID] [--num-envs NUM_ENVS] [--num-steps-collected NUM_STEPS_COLLECTED] [--num-mini-batches NUM_MINI_BATCHES] [--num-epochs NUM_EPOCHS] [--learning-rate LEARNING_RATE] [--total-timesteps TOTAL_TIMESTEPS]
                     [--gae-gamma GAE_GAMMA] [--gae-lambda GAE_LAMBDA] [--clip-coeff CLIP_COEFF] [--ent-loss-coeff ENT_LOSS_COEFF] [--value-loss-coeff VALUE_LOSS_COEFF] [--max-gradient MAX_GRADIENT] [--seed SEED] [--deterministic-torch] [--use-gpu]

PPO Atari Training Script

options:
  -h, --help            show this help message and exit
  --environment-id ENVIRONMENT_ID
                        environment ID (default: ALE/Assault-v5)
  --num-envs NUM_ENVS   number of environments (default: 8)
  --num-steps-collected NUM_STEPS_COLLECTED
                        number of steps collected per rollout (default: 128)
  --num-mini-batches NUM_MINI_BATCHES
                        number of mini-batches (default: 4)
  --num-epochs NUM_EPOCHS
                        number of epochs per training phase (default: 4)
  --learning-rate LEARNING_RATE
                        learning rate (default: 2.5e-4)
  --total-timesteps TOTAL_TIMESTEPS
                        total number of timesteps (default: 1e7)
  --gae-gamma GAE_GAMMA
                        GAE gamma parameter (default: 0.99)
  --gae-lambda GAE_LAMBDA
                        GAE lambda parameter (default: 0.95)
  --clip-coeff CLIP_COEFF
                        clipping coefficient (default: 0.1)
  --ent-loss-coeff ENT_LOSS_COEFF
                        entropy loss coefficient (default: 0.01)
  --value-loss-coeff VALUE_LOSS_COEFF
                        value loss coefficient (default: 0.5)
  --max-gradient MAX_GRADIENT
                        maximum gradient value (default: 0.5)
  --seed SEED           random seed (default: 1)
  --deterministic-torch
                        whether to set deterministic torch behavior (default: False)
  --use-gpu             whether to use GPU (default: False)

```

https://user-images.githubusercontent.com/26146888/225728444-7b762845-e32f-4dfb-814c-57a12459ed39.mp4

https://user-images.githubusercontent.com/26146888/237050171-ea751180-80f4-4b7f-af64-24bb08979b10.mp4

https://github.com/VerleysenNiels/PPO-pytorch-gym/assets/26146888/bf604044-d7f5-4477-ba4c-78bc3f461de9

https://github.com/VerleysenNiels/PPO-pytorch-gym/assets/26146888/05233fc2-b5ca-4a62-ba2c-93311aaff33e


## References

For more information on the Proximal Policy Optimization algorithm, see the original paper by Schulman et al. (2017): ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347). 

For more information on using reinforcement learning in OpenAI Gym, see the official documentation: ["Using Reinforcement Learning (RL) in OpenAI Gym"](https://gym.openai.com/docs/).
