from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable

EPS = 1e-8


class ReturnEstimator(ABC):
    """Abstract class for estimating returns based on some trajectories."""
    @abstractmethod
    def returns(self, *args, **kwargs):
        raise NotImplementedError


class ActorCritic(ReturnEstimator):
    """Class that encapsulates the advantage actor-critic algorithm.

    Args:
        gamma: Discount value
        normalise_returns: Whether or not to normalise target returns
    """
    def __init__(self,
                 gamma: float,
                 normalise_returns: bool = False,
                 use_gae: bool = False,
                 gae_lambda: float = None,
                 dtype: torch.dtype = torch.float):
        self.gamma = gamma
        self.normalise_returns = normalise_returns
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.dtype = dtype

    def returns(self,
                bootstrap_values: torch.Tensor,
                rewards: torch.Tensor,
                values: torch.Tensor,
                log_probs: torch.Tensor,
                dones: torch.Tensor):
        """Calculate Advantage actor-critic returns.

        Args:
            bootstrap_values: Vector containing estimated value of final states each trajectory.
                Shape (num_envs, 1)
            rewards: Rewards for trajectories. Shape: (num_envs, num_steps, 1)
            values: Values for trajectory states: Shape (num_envs, num_steps), 1
            log_probs: Log probabilities of actions taken during trajectory. Shape: (num_envs, num_steps, 1)
            dones: Done masks for trajectory states. Shape: (num_envs, num_steps, 1)
        """
        returns = []
        if self.use_gae:
            gae = 0
            for t in reversed(range(rewards.size(0))):
                if t == rewards.size(0) - 1:
                    delta = rewards[t] + self.gamma * bootstrap_values * (~dones[t]).to(self.dtype) - values[t]
                else:
                    delta = rewards[t] + self.gamma * values[t+1] * (~dones[t]).to(self.dtype) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (~dones[t]).to(self.dtype) * gae
                R = gae + values[t]
                returns.insert(0, R)
        else:
            R = bootstrap_values * (~dones[-1]).to(self.dtype)
            for r, d in zip(reversed(rewards), reversed(dones)):
                R = r + self.gamma * R * (~d).to(self.dtype)
                returns.insert(0, R)

        returns = torch.stack(returns)

        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)

        return returns