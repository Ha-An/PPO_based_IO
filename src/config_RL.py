import numpy as np
from gym import spaces
from config_SimPy import *

# PPO Hyperparameters
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # GAE parameter
CLIP_EPSILON = 0.2  # PPO clip parameter
CRITIC_DISCOUNT = 0.5  # Value function coefficient
ENTROPY_BETA = 0.01  # Entropy coefficient for exploration
LR_ACTOR = 0.0005  # Learning rate for actor
LR_CRITIC = 0.001  # Learning rate for critic
EPOCHS = 10  # Number of epochs to train on the collected data
BATCH_SIZE = 64  # Minibatch size

# Training Parameters
EPISODES = 10  # Number of episodes for training
MAX_STEPS_PER_EPISODE = SIM_TIME  # Max steps per episode (in days)
PPO_STEPS = 2048  # Steps per PPO update
UPDATE_INTERVAL = 20  # Days between model updates
SAVE_INTERVAL = 100  # Episodes between model saves
EVAL_INTERVAL = 100  # Episodes between evaluations (default: 100)
EVAL_EPISODES = 3  # Number of episodes for evaluation (default: 3)

# Environment Parameters
ACTION_MIN = 0  # Minimum order quantity
ACTION_MAX = 5  # Maximum order quantity

# Action Space Definition
ACTION_SPACE_SIZE = spaces.MultiDiscrete([ACTION_MAX + 1] * MAT_COUNT)

# State Space Definition
STATE_MINS = []
STATE_MAXS = []
# On-hand inventory levels for all items (materials, wips, products)
for _ in range(len(I)):
    STATE_MINS.append(INVEN_LEVEL_MIN)
    STATE_MAXS.append(INVEN_LEVEL_MAX)
# In-transition inventory levels for material items
for _ in range(MAT_COUNT):
    STATE_MINS.append(0)
    STATE_MAXS.append(ACTION_MAX * 7)  # Assuming maximum lead time of 7 days
# Remaining demand
STATE_MINS.append(0)
STATE_MAXS.append(DEMAND_SCENARIO["max"])
# Convert to numpy arrays
STATE_MINS = np.array(STATE_MINS, dtype=np.int32)
STATE_MAXS = np.array(STATE_MAXS, dtype=np.int32)
# Define state space
MULTI_STATE_SPACE_SIZE = spaces.MultiDiscrete(STATE_MAXS - STATE_MINS + 1)

# Network Parameters
HIDDEN_LAYER_SIZES = [128, 128]  # Sizes of hidden layers in neural networks

# File Paths
MODEL_DIR = 'models'  # Directory to save models
LOG_DIR = 'logs'  # Directory to save logs
