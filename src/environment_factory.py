import gym

# Atari preprocessing wrappers as defined in stable baselines
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

def create_env_factory(env_id, seed, index, run_name):
    def make_env():
        # Start by initializing the environment itself
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Record videos of the agent
        if index == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        # Set the seed in the environment
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Return the environment
        return env

    return make_env


def create_env_factory_atari(env_id, seed, index, run_name):
    def make_env():
        # Start by initializing the environment itself
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Record videos of the agent
        if index == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        
        # Atari wrappers for preprocessing
        env = NoopResetEnv(env, noop_max=30)                    # Sample starting points by taking random number of NOOPS
        env = MaxAndSkipEnv(env, skip=4)                        # Skip frames after every action to save compute
        env = EpisodicLifeEnv(env)                              # End of life = end of episode
        if "FIRE" in env.unwrapped.get_action_meanings():       # If FIRE action starts the environment, then do that action automatically
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)                                # Clips rewards within range of -1 to 1
        env = gym.wrappers.ResizeObservation(env, (84, 84))     # Resizes the images
        env = gym.wrappers.GrayScaleObservation(env)            # Convert images to grayscale
        env = gym.wrappers.FrameStack(env, 4)                   # Stacks frames together in order to enable the agent to understand temporal effects like movements better.
        
        # Set the seed in the environment
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Return the environment
        return env

    return make_env