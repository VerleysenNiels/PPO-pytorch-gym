# Imports
import argparse
import logging
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.environment_factory import create_env_factory_atari
from src.ppo_agents import AtariAgentSmall

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='PPO Atari Training Script')
    
    parser.add_argument('--environment-id', type=str, default='ALE/Assault-v5', help='environment ID')
    
    parser.add_argument('--num-envs', type=int, default=8, help='number of environments')
    parser.add_argument('--num-steps-collected', type=int, default=128, help='number of steps collected per rollout')
    parser.add_argument('--num-mini-batches', type=int, default=4, help='number of mini-batches')
    parser.add_argument('--num-epochs', type=int, default=4, help='number of epochs per training phase')

    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--total-timesteps', type=int, default=10000000, help='total number of timesteps')

    parser.add_argument('--gae-gamma', type=float, default=0.99, help='GAE gamma parameter')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')

    parser.add_argument('--clip-coeff', type=float, default=0.1, help='clipping coefficient')
    parser.add_argument('--ent-loss-coeff', type=float, default=0.01, help='entropy loss coefficient')
    parser.add_argument('--value-loss-coeff', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--max-gradient', type=float, default=0.5, help='maximum gradient value')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--deterministic-torch', action='store_true', help='whether to set deterministic torch behavior')
    parser.add_argument('--use-gpu', action='store_true', help='whether to use GPU')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic_torch

    # Use GPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # Generate run-name
    run_name = f"PPO-{args.environment_id}-{int(time.time())}"

    logging.info(f"Training PPO agent on {args.environment_id}.\n\tUsing {device}\n\tRun id: {run_name}")
    
    # Initialize tensorboard summary writer
    writer = SummaryWriter(f"tensorboard/{run_name}")

    # Initialize the environment
    # For training PPO we need to use a vectorized architecture, this means that a single agent will be learning from
    # multiple instances of the environment at the same time. First during a rollout phase, the agent plays for a number of
    # steps in each environment storing all the experiences. Then in the learning phase, the agent improves based on what it has seen.
    environments = gym.vector.SyncVectorEnv([create_env_factory_atari(args.environment_id, args.seed + i, i, run_name) for i in range(NUM_ENVS)])
    
    assert isinstance(environments.single_action_space, gym.spaces.Discrete), "PPO only supports environments with a discrete action space."
    
    # Init agent and optimizer
    agent = AtariAgentSmall(environments)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage of rollout experiences
    observations = torch.zeros((args.num_steps_collected, args.num_envs) + environments.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps_collected, args.num_envs) + environments.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps_collected, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps_collected, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps_collected, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps_collected, args.num_envs)).to(device)
    
    # Prepare for playing
    global_step = 0
    start_timestamp = time.time()
    next_observation = torch.Tensor(environments.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    batch_size = int(args.num_envs * args.num_steps_collected)
    num_updates_to_perform = args.total_timesteps // batch_size

    # You can update this to add more information in the progress bar during training
    description = {"loss":0.0}
    
    # Training loop
    for update_idx in tqdm(range(1, num_updates_to_perform + 1), desc="Training", postfix=description):
        # Learning rate decay (annealing)
        # The learning rate will gradually decay from the configured value in the first loop to 0 in the last.
        current_lr = args.learning_rate * (1 - (update_idx - 1) / num_updates_to_perform)
        optimizer.param_groups[0]["lr"] = current_lr
        
        # Rollout phase
        # Perform actions for a given number of steps in each of the environment instances. Collect all experiences.
        for step in range(args.num_steps_collected):
            global_step += args.num_envs
            
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
            for t in reversed(range(args.num_steps_collected)):
                if t == args.num_steps_collected - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + args.gae_gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = delta + args.gae_gamma * args.gae_lambda * nextnonterminal * next_advantage
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
        for epoch in range(args.num_epochs):
            # Random shuffling and division in minibatches
            np.random.shuffle(b_indices)
            for start in range(0, batch_size, args.num_mini_batches):
                end = start + args.num_mini_batches
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
                    clipped_fractions += [((ratio - 1.0).abs() > args.clip_coeff).float().mean().item()]

                # Get and normalize the advantages of this minibatch 
                mb_advantages = b_advantages[mb_indices]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) 

                # Clipped surrogate objective
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coeff, 1 + args.clip_coeff)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[mb_indices]) ** 2
                v_clipped = b_values[mb_indices] + torch.clamp(new_value - b_values[mb_indices], -args.clip_coeff, args.clip_coeff)
                v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - args.ent_loss_coeff * entropy_loss + args.value_loss_coeff * v_loss

                # Optimizer step with clipped gradients
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_gradient)
                optimizer.step()
        
        # Update loss for the progress bar
        description = {"loss":loss.item()}
                
        # Write results of this rollout and training phase to tensorboard.
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/approximate_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/mean_fraction_clipped", np.mean(clipped_fractions), global_step) 
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        
