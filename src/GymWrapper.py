import os
import time
import numpy as np
import torch
from typing import Tuple, Dict, Any, List, Optional

from config_RL import *
from GymEnvironment import InventoryManagementEnv
from ppo import PPO
from log_RL import Logger


class GymWrapper:
    """
    Wrapper class for training and evaluating PPO agent in Gym environment.
    """

    def __init__(self, env=None, agent=None, logger=None):
        """
        Initialize the wrapper.

        Args:
            env: The gym environment
            agent: The PPO agent
            logger: Logger for tracking metrics
        """
        self.env = env if env is not None else InventoryManagementEnv()
        self.agent = agent if agent is not None else PPO(
            self.env.observation_space.shape[0],
            self.env.action_space.nvec,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            lr_actor=LR_ACTOR,
            lr_critic=LR_CRITIC
        )
        self.logger = logger if logger is not None else Logger(LOG_DIR)

        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)

    def train(self, episodes: int = EPISODES) -> None:
        """
        Train the agent.

        Args:
            episodes (int): Number of episodes to train for
        """
        best_reward = float('-inf')

        for episode in range(1, episodes + 1):
            # print(f"\nStarting episode {episode}/{episodes}")
            start_time = time.time()

            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0

            # Collect trajectory data
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

            # # For Action Distribution Logging
            # action_distribution = [[] for _ in range(MAT_COUNT)]

            for step in range(MAX_STEPS_PER_EPISODE):
                # Select action
                action, log_prob, value = self.agent.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                # # Log action distribution
                # for i in range(MAT_COUNT):
                #     action_distribution[i].append(action)

                # Store data
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                episode_length += 1

                # Print info every few steps
                # if (step + 1) % 10 == 0:
                #     print(
                #         f"Step {step+1}/{MAX_STEPS_PER_EPISODE} - Reward so far: {episode_reward:.2f}")

                # Update agent periodically
                if (step + 1) % UPDATE_INTERVAL == 0 or done:
                    # print(f"Updating agent at step {step+1}...")
                    update_metrics = self.agent.update(
                        states, actions, rewards, dones, log_probs, values)
                    self.logger.log_update(episode, update_metrics)
                    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

                if done:
                    # print(f"Episode done at step {step+1}")
                    break

            # Calculate episode metrics
            episode_time = time.time() - start_time

            # # Log detailed inventory metrics
            # inventory_metrics = {
            #     f"inventory/{I[inven.item_id]['NAME']}": inven.on_hand_inventory
            #     for inven in self.env.inventoryList
            # }

            # # Add in-transit inventory for materials
            # inventory_metrics.update({
            #     f"in_transit/{I[inven.item_id]['NAME']}": inven.in_transition_inventory
            #     for inven in self.env.inventoryList if I[inven.item_id]['TYPE'] == 'Material'
            # })

            # Log episode results
            self.logger.log_episode(
                episode=episode,
                episode_reward=episode_reward,
                episode_length=episode_length,
                episode_time=episode_time,
                # custom_metrics=inventory_metrics,
                # action_distribution=action_distribution
            )

            print(
                f"Episode {episode}/{episodes} - Total Reward: {episode_reward:.2f} ")

            # Save model if it's the best so far
            if episode_reward > best_reward:
                best_reward = episode_reward
            #     self.agent.save(os.path.join(MODEL_DIR, "best_model.pt"))
            #     # print(
            #     #     f"New best model saved with reward: {best_reward:.2f} at episode {episode}")

            # Save model periodically
            # if episode % SAVE_INTERVAL == 0:
            #     self.agent.save(os.path.join(
            #         MODEL_DIR, f"model_episode_{episode}.pt"))
            #     # print(f"Model periodically saved at episode {episode}")

            # # Evaluate model periodically
            # if episode % EVAL_INTERVAL == 0:
            #     eval_reward = self.evaluate(episodes=EVAL_EPISODES)
            #     self.logger.log_eval(episode=episode, reward=eval_reward)
            #     print(
            #         f"Evaluation after {episode}/{episodes} - Average Reward: {eval_reward:.2f}")

    def evaluate(self, episodes: int = 1, render: bool = False) -> float:
        """
        Evaluate the agent.

        Args:
            episodes (int): Number of episodes to evaluate for
            render (bool): Whether to render the environment

        Returns:
            float: Average reward over all episodes
        """
        total_rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action deterministically (no exploration)
                action, _, _ = self.agent.select_action(
                    state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state

                if render:
                    self.env.render()

            total_rewards.append(episode_reward)

            if render:
                print(
                    f"Evaluation Episode {episode + 1} - Total Reward: {episode_reward:.2f}")

        return np.mean(total_rewards)

    def load_model(self, path: str) -> None:
        """
        Load a saved model.

        Args:
            path (str): Path to the saved model
        """
        self.agent.load(path)
        print(f"Model loaded from {path}")
