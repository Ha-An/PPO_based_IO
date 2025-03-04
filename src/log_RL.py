import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Optional


class Logger:
    """
    Logger for training metrics using TensorBoard.
    """

    def __init__(self, log_dir: str):
        """
        Initialize the logger.

        Args:
            log_dir (str): Directory to save logs
        """
        # Create a unique log directory with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.start_time = time.time()

        print(f"Logging to: {self.log_dir}")

    def log_episode(self, episode: int, episode_reward: float, episode_length: int, episode_time: float, action_distribution) -> None:
        """
        Log episode metrics.

        Args:
            episode (int): Episode number
            episode_reward (float): Episode reward
            episode_length (int): Episode length
            episode_time (float): Episode time
            custom_metrics (Dict[str, float]): Custom metrics to log
        """
        # Log standard metrics
        self.writer.add_scalar('train/episode_reward', episode_reward, episode)
        self.writer.add_scalar('train/episode_length', episode_length, episode)
        self.writer.add_scalar('train/simulation time', episode_time, episode)

        # # Calculate running stats
        # elapsed = time.time() - self.start_time
        # self.writer.add_scalar('train/total_elapsed_time', elapsed, episode)
        # self.writer.add_scalar('train/episodes_per_hour',
        #                        episode / (elapsed / 3600), episode)

        # # Log custom metrics
        # if custom_metrics is not None:
        #     for metric_name, metric_value in custom_metrics.items():
        #         self.writer.add_scalar(f'{metric_name}', metric_value, episode)

        # Action Distribution (히스토그램)
        if action_distribution is not None:
            print("action_distribution: ", action_distribution)
            for i, acts in enumerate(action_distribution):
                if len(acts) != 0:
                    # acts는 매 스텝마다 기록된 action이 들어있는 list
                    self.writer.add_histogram(
                        f'Actions/Order Qty for Material_{i}', acts, episode)

    def log_eval(self, episode: int, reward: float, custom_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log evaluation metrics.

        Args:
            episode (int): Episode number
            reward (float): Average evaluation reward
            custom_metrics (Dict[str, float]): Custom metrics to log
        """
        # Log evaluation reward
        self.writer.add_scalar('eval/reward', reward, episode)

        # Log custom metrics
        if custom_metrics:
            for metric_name, metric_value in custom_metrics.items():
                self.writer.add_scalar(
                    f'eval/{metric_name}', metric_value, episode)

    def log_update(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Log update metrics.

        Args:
            episode (int): Episode number
            metrics (Dict[str, float]): Update metrics
        """
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(
                f'update/{metric_name}', metric_value, episode)

    # def log_histogram(self, episode: int, tag: str, values: np.ndarray) -> None:
    #     """
    #     Log histogram of values.

    #     Args:
    #         episode (int): Episode number
    #         tag (str): Tag for the histogram
    #         values (np.ndarray): Values to log
    #     """
    #     if len(values) == 0:
    #         print(f"Warning: Empty values for histogram {tag}")
    #         return

    #     # Log the histogram
    #     self.writer.add_histogram(tag, values, episode)

    #     # Also log some statistics for this distribution
    #     self.writer.add_scalar(f"{tag}/mean", np.mean(values), episode)
    #     self.writer.add_scalar(f"{tag}/std", np.std(values), episode)
    #     self.writer.add_scalar(f"{tag}/min", np.min(values), episode)
    #     self.writer.add_scalar(f"{tag}/max", np.max(values), episode)

    # def log_inventory_stats(self, episode: int, inventories: Dict[str, float], in_transit: Dict[str, float] = None) -> None:
    #     """
    #     Log detailed inventory statistics.

    #     Args:
    #         episode (int): Episode number
    #         inventories (Dict[str, float]): Dictionary of inventory levels
    #         in_transit (Dict[str, float]): Dictionary of in-transit inventory levels
    #     """
    #     for item_name, level in inventories.items():
    #         self.writer.add_scalar(f'inventory/{item_name}', level, episode)

    #     if in_transit:
    #         for item_name, level in in_transit.items():
    #             self.writer.add_scalar(
    #                 f'in_transit/{item_name}', level, episode)

    # def log_action_distribution(self, episode: int, actions: List[int], material_names: List[str]) -> None:
    #     """
    #     Log distribution of actions taken for each material.

    #     Args:
    #         episode (int): Episode number
    #         actions (List[int]): Actions taken
    #         material_names (List[str]): Names of materials
    #     """
    #     for i, name in enumerate(material_names):
    #         self.writer.add_scalar(f'actions/{name}', actions[i], episode)

    def close(self) -> None:
        """
        Close the logger.
        """
        self.writer.close()
        print(f"Logger closed. Log saved to: {self.log_dir}")
