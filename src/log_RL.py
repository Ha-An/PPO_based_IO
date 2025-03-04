from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Logger:
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_overfitting_metrics(self, train_loss, val_loss, step):
        self.writer.add_scalars("Loss/Train_vs_Val",
                                {"Train": train_loss, "Val": val_loss}, step)

    def close(self):
        self.writer.close()


# import os
# import time
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from typing import Dict, Any, List, Optional


# class Logger:
#     """
#     Logger for training metrics using TensorBoard.
#     """

#     def __init__(self, log_dir: str):
#         """
#         Initialize the logger.

#         Args:
#             log_dir (str): Directory to save logs
#         """
#         # Create a unique log directory with timestamp
#         timestamp = time.strftime("%Y%m%d-%H%M%S")
#         self.log_dir = os.path.join(log_dir, f"run_{timestamp}")
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.writer = SummaryWriter(log_dir=self.log_dir)
#         self.start_time = time.time()

#         print(f"Logging to: {self.log_dir}")

#     def log_episode(self, episode: int, reward: float, length: int, time: float, custom_metrics: Optional[Dict[str, float]] = None) -> None:
#         """
#         Log episode metrics.

#         Args:
#             episode (int): Episode number
#             reward (float): Episode reward
#             length (int): Episode length
#             time (float): Episode time
#             custom_metrics (Dict[str, float]): Custom metrics to log
#         """
#         # Log standard metrics
#         self.writer.add_scalar('train/reward', reward, episode)
#         self.writer.add_scalar('train/length', length, episode)
#         self.writer.add_scalar('train/time', time, episode)

#         # Calculate running stats
#         elapsed = time.time() - self.start_time
#         self.writer.add_scalar('train/total_elapsed_time', elapsed, episode)
#         self.writer.add_scalar('train/episodes_per_hour',
#                                episode / (elapsed / 3600), episode)

#         # Log custom metrics
#         if custom_metrics:
#             for metric_name, metric_value in custom_metrics.items():
#                 self.writer.add_scalar(f'{metric_name}', metric_value, episode)

#     def log_eval(self, episode: int, reward: float, custom_metrics: Optional[Dict[str, float]] = None) -> None:
#         """
#         Log evaluation metrics.

#         Args:
#             episode (int): Episode number
#             reward (float): Average evaluation reward
#             custom_metrics (Dict[str, float]): Custom metrics to log
#         """
#         # Log evaluation reward
#         self.writer.add_scalar('eval/reward', reward, episode)

#         # Log custom metrics
#         if custom_metrics:
#             for metric_name, metric_value in custom_metrics.items():
#                 self.writer.add_scalar(
#                     f'eval/{metric_name}', metric_value, episode)

#     def log_update(self, episode: int, metrics: Dict[str, float]) -> None:
#         """
#         Log update metrics.

#         Args:
#             episode (int): Episode number
#             metrics (Dict[str, float]): Update metrics
#         """
#         for metric_name, metric_value in metrics.items():
#             self.writer.add_scalar(
#                 f'update/{metric_name}', metric_value, episode)

#     def log_histogram(self, episode: int, tag: str, values: np.ndarray) -> None:
#         """
#         Log histogram of values.

#         Args:
#             episode (int): Episode number
#             tag (str): Tag for the histogram
#             values (np.ndarray): Values to log
#         """
#         self.writer.add_histogram(tag, values, episode)

#     def log_inventory_stats(self, episode: int, inventories: Dict[str, float], in_transit: Dict[str, float] = None) -> None:
#         """
#         Log detailed inventory statistics.

#         Args:
#             episode (int): Episode number
#             inventories (Dict[str, float]): Dictionary of inventory levels
#             in_transit (Dict[str, float]): Dictionary of in-transit inventory levels
#         """
#         for item_name, level in inventories.items():
#             self.writer.add_scalar(f'inventory/{item_name}', level, episode)

#         if in_transit:
#             for item_name, level in in_transit.items():
#                 self.writer.add_scalar(
#                     f'in_transit/{item_name}', level, episode)

#     def log_action_distribution(self, episode: int, actions: List[int], material_names: List[str]) -> None:
#         """
#         Log distribution of actions taken for each material.

#         Args:
#             episode (int): Episode number
#             actions (List[int]): Actions taken
#             material_names (List[str]): Names of materials
#         """
#         for i, name in enumerate(material_names):
#             self.writer.add_scalar(f'actions/{name}', actions[i], episode)

#     def close(self) -> None:
#         """
#         Close the logger.
#         """
#         self.writer.close()
#         print(f"Logger closed. Log saved to: {self.log_dir}")
