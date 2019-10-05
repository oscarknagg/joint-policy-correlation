from abc import ABC, abstractmethod
from typing import List


class MultiAgentTrainer(ABC):
    """Abstract class for training multiple agents."""
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError


class SingleAgentTrainer(ABC):
    """Abstract class for training a single agent."""
    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError


class IndependentTrainer(MultiAgentTrainer):
    """Trains multiple agents independently.

    Under this training regime each agent considers the other agents
    as part of the environment and no multi-agent modifications to
    standard RL training methods are made.
    """
    def __init__(self, trainers: List[SingleAgentTrainer]):
        self.trainers = trainers

    def train(self, *args, **kwargs):
        for tr in self.trainers:
            tr.train(*args, **kwargs)
