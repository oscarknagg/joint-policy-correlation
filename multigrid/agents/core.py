from abc import ABC, abstractmethod
from torch import nn


class RLAgent(ABC):
    agent_id: int = 0
    pool_id: int = 0
    train_steps: int = 0
    train_episodes: int = 0
    recurrent: str = 0
