import gym

def create_env_factory(env_id, seed):
    def make_env():
        # Start by initializing the environment itself
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # Set the seed in the environment
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        
        # Return the environment
        return env

    return make_env