# System Architecture Overview


## Environment (GymEnvironment.py):

* Wraps the SimPy environment into a standard Gym interface
* Defines state space with inventory levels, in-transit items, and demand
* Takes actions (order quantities) and returns rewards (-cost)
Normalizes states for better neural network learning


## PPO Implementation (ppo.py):

* Implements the PPO algorithm for optimizing inventory decisions
* Uses actor-critic architecture with separate networks for policy and value functions
* Supports multi-discrete action spaces for ordering multiple materials
* Includes gradient clipping, advantage normalization, and entropy regularization


## GymWrapper (GymWrapper.py):

* Handles training and evaluation loops
* Collects trajectories and triggers PPO updates
* Tracks performance metrics and saves models


## Logging (log_RL.py):

* Uses TensorBoard for visualization of training performance
* Tracks rewards, costs, inventory levels, and agent behavior
* Creates timestamped log directories for experiment tracking


## Configuration (config_RL.py):

* Contains all RL hyperparameters including learning rates, discount factors
* Defines state and action spaces
* Specifies network architecture and training parameters


## Main Script (main.py):

* Entry point with command-line arguments for different modes
* Supports training, evaluation, and visualization
* Sets random seeds for reproducibility

# Running the System

## For training
`
python main.py --mode train --episodes 1000
`

## For evaluation
`
python main.py --mode evaluate --episodes 10 --model_path models/best_model.pt
`
## For visualization with rendering
`
python main.py --mode visualize --model_path models/best_model.pt --render
`