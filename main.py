# Imports
import random
import time
import gym
import numpy as np
import torch
import torch.optim as optim

from src.environment_factory import create_env_factory
from src.ppo_agents import AgentSmall

# Arguments (TODO: handle with argument parser)
SEED = 1
DETERMINISTIC_TORCH = True
USE_GPU = True

ENVIRONMENT_ID = "CartPole-v1" # Start with cartpole for development
NUM_ENVS = 4
NUM_STEPS_COLLECTED = 128

LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 25000

GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95

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
    
    assert isinstance(environments.single_action_space, gym.spaces.Discrete), "PPO only supports environments with a discrete action space."
    
    # Init agent and optimizer
    agent = AgentSmall(environments)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Storage of rollout experiences
    observations = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS) + environments.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS) + environments.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS_COLLECTED, NUM_ENVS)).to(device)
    
    # Prepare for playing
    global_step = 0
    start_timestamp = time.time()
    next_observation = environments.reset()
    next_done = torch.zeros(NUM_ENVS)
    
    batch_size = int(NUM_ENVS * NUM_STEPS_COLLECTED)
    num_updates_to_perform = TOTAL_TIMESTEPS // batch_size
    
    # Training loop
    for update_idx in range(1, num_updates_to_perform + 1):
        # Learning rate decay (annealing)
        # The learning rate will gradually decay from the configured value in the first loop to 0 in the last.
        current_lr = LEARNING_RATE * (1 - (update_idx - 1) / num_updates_to_perform)
        optimizer.param_groups[0]["lr"] = current_lr
        
        # Rollout phase
        # Perform actions for a given number of steps in each of the environment instances. Collect all experiences.
        for step in range(NUM_STEPS_COLLECTED):
            global_step += NUM_ENVS
            
            # Store state
            observations[step] = next_observation
            dones[step] = next_done
            
            # Select and store action
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_observation)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Perform action and store result
            next_observation, reward, done, info = environments.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_observation = torch.Tensor(next_observation).to(device)
            next_done = torch.Tensor(done).to(device)
        
        # Generalized Advantage Estimation (GAE)
        # Compute advantages for training    
        with torch.no_grad():
            # Estimate the values of the next states if not done yet
            next_value = agent.get_value(next_observation).reshape(1, -1)
            # Init datastructure to store advantages
            advantages = torch.zeros_like(rewards).to(device)
            # Advantage needs to take into account the advantage in the next state as well.
            # As we start at the end, we initialize this to 0 and afterwards just store the previous advantage estimate in this variable.
            next_advantage = 0
            
            # Actual advantage estimation by going from end to start over the collected observations.
            for t in reversed(range(NUM_STEPS_COLLECTED)):
                if t == NUM_STEPS_COLLECTED - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + GAE_GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = delta + GAE_GAMMA * GAE_LAMBDA * nextnonterminal * last_gae
                last_gae = advantages[t]
                
            returns = advantages + values
            