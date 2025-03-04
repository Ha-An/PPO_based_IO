import argparse
import os
import torch
import numpy as np
import random
import time
from config_SimPy import *
from config_RL import *
from GymEnvironment import InventoryManagementEnv
from GymWrapper import GymWrapper
from ppo import PPO
from log_RL import Logger


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='PPO for Inventory Management')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'visualize'],
                        help='Mode to run in (train, evaluate, visualize)')
    parser.add_argument('--episodes', type=int,
                        default=EPISODES, help='Number of episodes to run')
    parser.add_argument('--model_path', type=str,
                        default=None, help='Path to load model from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment during evaluation')
    parser.add_argument('--lr_actor', type=float,
                        default=LR_ACTOR, help='Learning rate for actor')
    parser.add_argument('--lr_critic', type=float,
                        default=LR_CRITIC, help='Learning rate for critic')
    parser.add_argument('--batch_size', type=int,
                        default=BATCH_SIZE, help='Batch size for training')

    return parser.parse_args()


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def main():
    """
    Main function.
    """
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create environment
    env = InventoryManagementEnv()

    # Print environment details
    print("\n=== Inventory Management Environment ===")
    print(f"Items: {len(I)} (Products: {len([i for i in I.values() if i['TYPE'] == 'Product'])}, "
          f"Materials: {MAT_COUNT}, "
          f"WIPs: {len([i for i in I.values() if i['TYPE'] == 'WIP'])})")
    print(f"Processes: {len(P)}")
    print(f"Demand Scenario: {DEMAND_SCENARIO}")
    print(f"Lead Time Scenario: {LEADTIME_SCENARIO}")
    print(f"Simulation Time: {SIM_TIME} days")
    print("=======================================\n")

    # Print RL configuration
    print("=== Reinforcement Learning Configuration ===")
    print(f"Action Space: {env.action_space}")
    print(f"Action Dimensions: {env.action_space.nvec}")
    print(f"State Space: {env.observation_space}")
    print(f"State Dimensions: {len(env.observation_space.nvec)}")
    print(f"Discount Factor (γ): {GAMMA}")
    print(
        f"PPO Parameters - Clip: {CLIP_EPSILON}, Entropy Beta: {ENTROPY_BETA}")
    print(f"Learning Rates - Actor: {args.lr_actor}, Critic: {args.lr_critic}")
    print("===========================================\n")

    # Create PPO agent
    agent = PPO(
        # env.observation_space.shape[0],
        len(env.observation_space.nvec),
        env.action_space.nvec,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        batch_size=args.batch_size
    )

    # Create logger
    logger = Logger(LOG_DIR)

    # Create wrapper
    wrapper = GymWrapper(env=env, agent=agent, logger=logger)

    # Load model if provided
    if args.model_path is not None:
        wrapper.load_model(args.model_path)

    # Run in specified mode
    if args.mode == 'train':
        print(f"\nStarting training for {args.episodes} episodes...")
        wrapper.train(episodes=args.episodes)
    elif args.mode == 'evaluate':
        print(f"\nStarting evaluation for {args.episodes} episodes...")
        avg_reward = wrapper.evaluate(
            episodes=args.episodes, render=args.render)
        print(f"Average Reward: {avg_reward:.2f}")
    elif args.mode == 'visualize':
        print("\nStarting visualization...")
        wrapper.evaluate(episodes=1, render=True)

    # Close logger
    logger.close()


# Start timing the computation
com_start_time = time.time()

if __name__ == "__main__":
    main()

# Calculate computation time and print it
com_end_time = time.time()
print(
    f"\nTotal computation time: {(com_end_time - com_start_time)/60:.2f} minutes")

# tensorboard --logdir=logs
