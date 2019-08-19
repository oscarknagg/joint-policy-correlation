from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution


class Interaction(NamedTuple):
    """Representation of a single step of interaction between a set of agents and an environment."""
    observations: Optional[Dict[str, Tensor]]
    action_distributions: Optional[Dict[str, Distribution]]
    actions: Optional[Dict[str, Tensor]]
    state_values: Optional[Dict[str, Tensor]]
    q_values: Optional[Dict[str, Tensor]]
    log_probs: Optional[Dict[str, Tensor]]


class InteractionHandler(ABC):
    """Interface for interaction of multiple agents with an environment."""
    @abstractmethod
    def interact(self, observations: Dict[str, Tensor], cx: Dict[str, Tensor], hx: Dict[str, Tensor]) -> (Interaction, Dict[str, Tensor], Dict[str, Tensor]):
        """


        Args:
            observations:
            cx:
            hx:

        Returns:
            interaction:
        """
        raise NotImplementedError


class ActionSampler(ABC):
    """Abstract for sampling method from probabilities."""
    @abstractmethod
    def sample(self, actions: Distribution) -> Tensor:
        raise NotImplementedError


class StochasticActionSampler(ActionSampler):
    def sample(self, actions: Distribution) -> Tensor:
        return actions.sample().clone().long()


class DeterministcActionSampler(ActionSampler):
    pass


class AgentPoolHandler(InteractionHandler):
    def __init__(self, models: Dict[str, nn.Module], keep_obs: bool):
        self.models = models
        self.keep_obs = keep_obs

    def interact(self,
                 observations: Dict[str, Tensor],
                 hx: Optional[Dict[str, Tensor]],
                 cx: Optional[Dict[str, Tensor]]) -> (Interaction, Dict[str, Tensor], Dict[str, Tensor]):
        action_distributions = {}
        actions = {}
        values = {}
        log_probs = {}
        for i, (agent, obs) in enumerate(observations.items()):
            model = self.models[agent]
            if model.recurrent == 'lstm':
                probs_, value_, hx[agent], cx[agent] = model(obs, hx[agent], cx[agent])
            elif model.recurrent == 'gru':
                probs_, value_, hx[agent] = model(obs, hx[agent])
            else:
                probs_, value_ = model(obs)

            action_distributions[agent] = Categorical(probs_)
            actions[agent] = action_distributions[agent].sample().clone().long()
            values[agent] = value_
            log_probs[agent] = action_distributions[agent].log_prob(actions[agent].clone())

        interaction = Interaction(
            observations=observations if self.keep_obs else None,
            action_distributions=action_distributions,
            actions=actions,
            state_values=values,
            q_values=None,
            log_probs=log_probs
        )

        return interaction, hx, cx


class MultiSpeciesHandler(InteractionHandler):
    """Multiple species as models with unshared weights."""
    def __init__(self, models: List[nn.Module], n_agents: int, keep_obs: bool):
        self.models = models
        self.n_agents = n_agents
        self.keep_obs = keep_obs

    def interact(self,
                 observations: Dict[str, Tensor],
                 hx: Optional[Dict[str, Tensor]],
                 cx: Optional[Dict[str, Tensor]]) -> (Interaction, Dict[str, Tensor], Dict[str, Tensor]):
        action_distributions = {}
        actions = {}
        values = {}
        log_probs = {}
        for i, (agent, obs) in enumerate(observations.items()):
            model = self.models[i]
            if model.recurrent == 'lstm':
                probs_, value_, hx[agent], cx[agent] = model(obs, hx[agent], cx[agent])
            elif model.recurrent == 'gru':
                probs_, value_, hx[agent] = model(obs, hx[agent])
            else:
                probs_, value_ = model(obs)

            action_distributions[agent] = Categorical(probs_)
            actions[agent] = action_distributions[agent].sample().clone().long()
            values[agent] = value_
            log_probs[agent] = action_distributions[agent].log_prob(actions[agent].clone())

        interaction = Interaction(
            observations=observations if self.keep_obs else None,
            action_distributions=action_distributions,
            actions=actions,
            state_values=values,
            q_values=None,
            log_probs=log_probs
        )

        return interaction, hx, cx
