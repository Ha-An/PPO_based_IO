import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any

from config_RL import *


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO algorithm.

    Consists of shared backbone layers followed by separate actor (policy) and
    critic (value) networks.
    """

    def __init__(self, state_dim, action_dims: List[int], hidden_layer_sizes: List[int] = [64, 64]):
        """
        Initialize the network.

        Args:
            state_dim (int): Dimension of the state space
            action_dims (List[int]): Dimensions of the action space for each material
            hidden_layer_sizes (List[int]): Sizes of hidden layers
        """
        super(ActorCritic, self).__init__()

        # Actor network (policy)
        actor_layers = []
        input_dim = state_dim

        # print(f"State dim: {state_dim}")
        # print(f"Action_dims: {action_dims}")

        for hidden_size in hidden_layer_sizes:
            actor_layers.append(nn.Linear(input_dim, hidden_size))
            actor_layers.append(nn.ReLU())
            input_dim = hidden_size

        self.actor_backbone = nn.Sequential(*actor_layers)

        # Action heads for multi-discrete actions
        # Each material has its own action head
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_layer_sizes[-1], dim) for dim in action_dims
        ])

        # Critic network (value function)
        critic_layers = []
        input_dim = state_dim

        for hidden_size in hidden_layer_sizes:
            critic_layers.append(nn.Linear(input_dim, hidden_size))
            critic_layers.append(nn.ReLU())
            input_dim = hidden_size

        critic_layers.append(nn.Linear(hidden_layer_sizes[-1], 1))

        self.critic = nn.Sequential(*critic_layers)

    def forward(self, state: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor]: Action distributions and value
        """
        actor_features = self.actor_backbone(state)
        action_dists = [F.softmax(head(actor_features), dim=-1)
                        for head in self.action_heads]
        value = self.critic(state)

        return action_dists, value

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get the value for a state.

        Args:
            state (torch.Tensor): State tensor

        Returns:
            torch.Tensor: Value
        """
        return self.critic(state)

    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, and value for a state.

        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): Whether to select actions deterministically

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Action, log probability, and value
        """
        action_dists, value = self(state)

        actions = []
        log_prob_sum = 0.0

        for dist in action_dists:
            # Handle distribution shape: ensure it's the right format for action selection
            if dist.dim() == 1:  # If dist is [action_dim]
                dist_to_use = dist.unsqueeze(0)  # Make it [1, action_dim]
            else:
                dist_to_use = dist

            if deterministic:
                action = torch.argmax(dist_to_use, dim=-1)
            else:
                action = torch.multinomial(dist_to_use, 1).squeeze(-1)

            # Handle different action shapes
            if action.dim() == 0:  # If action is a scalar
                action_idx = action.item()
                # Use first batch item if dist is batched
                prob = dist_to_use[0, action_idx]
            else:  # If action is batched [batch_size]
                batch_indices = torch.arange(len(action), device=action.device)
                prob = dist_to_use[batch_indices, action]

            log_prob = torch.log(prob + 1e-10)
            actions.append(action)
            log_prob_sum += log_prob

        # Stack actions along a new dimension
        return torch.stack(actions), log_prob_sum, value


class PPO:
    """
    Proximal Policy Optimization algorithm implementation.
    """

    def __init__(
        self,
        state_dim,
        action_dims: List[int],
        hidden_layer_sizes: List[int] = [64, 64],
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = GAMMA,
        gae_lambda: float = LAMBDA,
        clip_epsilon: float = CLIP_EPSILON,
        critic_discount: float = CRITIC_DISCOUNT,
        entropy_beta: float = ENTROPY_BETA,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE
    ):
        """
        Initialize the PPO agent.

        """
        self.actor_critic = ActorCritic(
            state_dim, action_dims, hidden_layer_sizes)

        # Separate optimizers for actor and critic
        actor_params = list(self.actor_critic.actor_backbone.parameters(
        )) + list(self.actor_critic.action_heads.parameters())
        critic_params = list(self.actor_critic.critic.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr_critic)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_discount = critic_discount
        self.entropy_beta = entropy_beta
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic.to(self.device)

        print(f"PPO initialized with device: {self.device}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select an action given a state.

        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to select actions deterministically

        Returns:
            Tuple[np.ndarray, float, float]: Action, log probability, and value
        """
        # print("State: ", state)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # print("state_tensor: ", state_tensor)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action_and_value(
                state_tensor, deterministic)
        # print("Action: ", action)

        # Ensure log_prob is a scalar
        if log_prob.dim() > 0:
            log_prob = log_prob.sum()  # Sum all log probabilities if necessary

        # Convert action tensor to a simple numpy array
        action_numpy = action.cpu().numpy()
        # print("action_numpy: ", action_numpy)

        # Make sure we return a 1D array where each element is the order quantity for a material
        if action_numpy.ndim > 1:
            action_numpy = action_numpy.flatten()

        return action_numpy, log_prob.cpu().item(), value.cpu().item()

    def update(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool],
        log_probs: List[float],
        values: List[float]
    ) -> Dict[str, float]:
        """
        Update the agent using PPO.

        Args:
            states (List[np.ndarray]): List of states
            actions (List[np.ndarray]): List of actions
            rewards (List[float]): List of rewards
            dones (List[bool]): List of done flags
            log_probs (List[float]): List of log probabilities
            values (List[float]): List of values

        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        # If no data to update on, return empty metrics
        if len(states) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(
            np.array(log_probs)).to(self.device)

        # Compute returns and advantages
        returns, advantages = self._compute_gae(rewards, values, dones)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages
        if len(advantages_tensor) > 1:  # Only normalize if we have more than one sample
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "total_loss": 0
        }

        for _ in range(self.epochs):
            # Create minibatches
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                # Forward pass
                action_dists, values = self.actor_critic(batch_states)

                # Calculate log probabilities
                log_probs = []
                entropy = 0

                for i, dist in enumerate(action_dists):
                    action = batch_actions[:, i].unsqueeze(-1)
                    log_prob = torch.log(dist.gather(
                        1, action) + 1e-10).squeeze(-1)
                    log_probs.append(log_prob)
                    entropy += -(dist * torch.log(dist + 1e-10)
                                 ).sum(dim=-1).mean()

                log_probs = torch.stack(log_probs).sum(dim=0)

                # PPO loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Total loss
                loss = policy_loss + self.critic_discount * \
                    value_loss - self.entropy_beta * entropy

                # Update metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["total_loss"] += loss.item()

                # Gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), max_norm=0.5)

                self.actor_optimizer.step()
                self.critic_optimizer.step()

        # Calculate average metrics
        total_batches = max(1, (len(states) // self.batch_size) * self.epochs)
        for key in metrics:
            metrics[key] /= total_batches

        return metrics

    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using Generalized Advantage Estimation.

        Args:
            rewards (List[float]): List of rewards
            values (List[float]): List of values
            dones (List[bool]): List of done flags

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns and advantages
        """
        n = len(rewards)
        if n == 0:
            return np.array([]), np.array([])

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)

        # Add a final value if the episode is not done
        next_value = 0 if dones[-1] else values[-1]

        # Compute GAE
        gae = 0
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - float(dones[t])
            next_value_step = next_value if t == n - 1 else values[t + 1]

            delta = rewards[t] + self.gamma * \
                next_value_step * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    def save(self, path: str) -> None:
        """
        Save the model.

        Args:
            path (str): Path to save the model
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load a saved model.

        Args:
            path (str): Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(
            checkpoint['actor_critic_state_dict'])
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict'])
