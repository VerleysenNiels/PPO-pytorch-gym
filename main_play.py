# Imports
import random
import gym
import numpy as np
import torch

# Arguments (TODO: handle with argument parser)
SEED = 1
DETERMINISTIC_TORCH = True
USE_GPU = True

if __name__ == "__main__"
# SEEDING
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = DETERMINISTIC_TORCH

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

# Initialize the environment