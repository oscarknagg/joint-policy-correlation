from abc import ABC, abstractmethod
from torch import nn


class RLAgent(ABC):
    agent_id: int
    pool_id: int
    train_steps: int
    train_episodes: int
    recurrent: str
