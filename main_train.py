# Imports
import random
import time
import gym
import numpy as np
import torch

from src.environment_factory import create_env_factory

# Arguments (TODO: handle with argument parser)
SEED = 1
DETERMINISTIC_TORCH = True
USE_GPU = True

ENVIRONMENT_ID = "CartPole-v1" # Start with cartpole for development
NUM_ENVS = 1


if __name__ == "__main__":
    # SEEDING
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = DETERMINISTIC_TORCH

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

    # Generate run-name
    run_name = f"PPO-{ENVIRONMENT_ID}-{int(time.time())}"

    # Initialize the environment
    # For training PPO we need to use a vectorized architecture, this means that a single agent will be learning from
    # multiple instances of the environment at the same time. First during a rollout phase, the agent plays for a number of
    # steps in each environment storing all the experiences. Then in the learning phase, the agent improves based on what it has seen.
    environments = gym.vector.SyncVectorEnv([create_env_factory(ENVIRONMENT_ID, SEED + i, i, run_name) for i in range(NUM_ENVS)])
    
    # Reset environment
    observation = environments.reset()
    
    # Test gameplaying loop with random actions
    for _ in range(200):
        action = environments.action_space.sample()
        observation, reward, done, info = environments.step(action)
        