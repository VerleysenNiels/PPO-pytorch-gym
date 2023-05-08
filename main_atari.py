# Imports
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pyvirtualdisplay import Display
from torch.utils.tensorboard import SummaryWriter

from src.environment_factory import create_env_factory_atari
from src.ppo_agents import AtariAgentSmall

# Arguments (TODO: handle with argument parser)
SEED = 1
DETERMINISTIC_TORCH = True
USE_GPU = True

ENVIRONMENT_ID = "ALE/KungFuMaster-v5"
NUM_ENVS = 8
NUM_STEPS_COLLECTED = 128
NUM_MINI_BATCHES = 4
NUM_EPOCHS = 4          # Number of epochs per training phase

LEARNING_RATE = 2.5e-4
TOTAL_TIMESTEPS = 10000000

GAE_GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_COEFF = 0.1
ENT_LOSS_COEFF = 0.01
VALUE_LOSS_COEFF = 0.5
MAX_GRADIENT = 0.5

if __name__ == "__main__":
    # Set up virtual display, making sure this code runs even if no displays are available
    #virtual_display = Display(visible=0, size=(1400, 900))
    #virtual_display.start()
    
    # SEEDING
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = DETERMINISTIC_TORCH

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")
    print(device)

    # Generate run-name
    run_name = f"PPO-{ENVIRONMENT_ID}-{int(time.time())}"
    
    # Initialize tensorboard summary writer
    writer = SummaryWriter(f"tensorboard/{run_name}")

    # Initialize the environment
    # For training PPO we need to use a vectorized architecture, this means that a single agent will be learning from
    # multiple instances of the environment at the same time. First during a rollout phase, the agent plays for a number of
    # steps in each environment storing all the experiences. Then in the learning phase, the agent improves based on what it has seen.
    environments = gym.vector.SyncVectorEnv([create_env_factory_atari(ENVIRONMENT_ID, SEED + i, i, run_name) for i in range(NUM_ENVS)])
    
    assert isinstance(environments.single_action_space, gym.spaces.Discrete), "PPO only supports environments with a discrete action space."
    
    # Init agent and optimizer
    agent = AtariAgentSmall(environments)
    agent.to(device)
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
    next_observation = torch.Tensor(environments.reset()).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    
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
                advantages[t] = delta + GAE_GAMMA * GAE_LAMBDA * nextnonterminal * next_advantage
                next_advantage = advantages[t]
                
            returns = advantages + values
            
        # Flattening of the data in the batch
        b_observations = observations.reshape((-1,) + environments.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + environments.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Policy and value network optimization
        b_indices = np.arange(batch_size)
        clipped_fractions = []
        # Optimize with the observations from the rollout for the configured number of epochs
        for epoch in range(NUM_EPOCHS):
            # Random shuffling and division in minibatches
            np.random.shuffle(b_indices)
            for start in range(0, batch_size, NUM_MINI_BATCHES):
                end = start + NUM_MINI_BATCHES
                mb_indices = b_indices[start:end]

                # Get predictions by the agent on the minibatch
                _, newlogprob, entropy, new_value = agent.get_action_and_value(b_observations[mb_indices], b_actions.long()[mb_indices])
                # Compare how the model has already evolved since the collection of this experience during rollout
                logratio = newlogprob - b_logprobs[mb_indices]
                ratio = logratio.exp()

                # DEBUG VARIABLES
                with torch.no_grad():
                    # Approximate Kullback–Leibler divergence to monitor how aggressive the policy is updated
                    approx_kl = ((ratio - 1) - logratio).mean()
                    # Monitor how often clipping is triggered
                    clipped_fractions += [((ratio - 1.0).abs() > CLIP_COEFF).float().mean().item()]

                # Get and normalize the advantages of this minibatch 
                mb_advantages = b_advantages[mb_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) 

                # Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEFF, 1 + CLIP_COEFF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb_indices]) ** 2
                v_clipped = b_values[mb_indices] + torch.clamp(new_value - b_values[mb_indices], -CLIP_COEFF, CLIP_COEFF)
                v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - ENT_LOSS_COEFF * entropy_loss + VALUE_LOSS_COEFF * v_loss

                # Optimizer step with clipped gradients
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRADIENT)
                optimizer.step()
                
        # Write results of this rollout and training phase to tensorboard.
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/approximate_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/mean_fraction_clipped", np.mean(clipped_fractions), global_step) 
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        
