import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence, Distribution
from typing import List, Dict, Optional

from multigrid.core import Callback


class DiversityLoss(Callback):
    def __init__(self, env, models: List[nn.Module], agent: str):
        super(DiversityLoss, self).__init__()
        self.env = env
        self.models = models
        self.agent = agent
        self.i = int(agent.split('_')[-1])
        self.mask_env_dones = False
        self.mask_agent_dones = True

        self.hidden_states = {
            f'agent_{i}': torch.zeros((self.env.num_envs, self.models[i].hidden_size), device=self.env.device)
            for i, _ in enumerate(self.models)
        }
        self.cell_states = {
            f'agent_{i}': torch.zeros((self.env.num_envs, self.models[i].cell_size), device=self.env.device)
            for i, _ in enumerate(self.models)
        }

    def before_step(self,
                    logs: Optional[dict] = None,
                    actions: Optional[Dict[str, torch.Tensor]] = None,
                    action_distributions: Optional[Dict[str, Distribution]] = None,
                    obs: Optional[Dict[str, torch.Tensor]] = None):
        """Update hidden states, get action distribution for the shadowing agents."""
        observations = obs[self.agent]

        shadow_distributions = {}
        for i, model in enumerate(self.models):
            agent = f'agent_{i}'
            with torch.no_grad():
                if model.recurrent == 'lstm':
                    probs_, value_, self.hidden_states[agent], self.cell_states[agent] = model(
                        observations, self.hidden_states[agent], self.cell_states[agent])
                elif model.recurrent == 'gru':
                    probs_, value_, self.hidden_states[agent] = model(observations, self.hidden_states[agent])
                else:
                    probs_, value_ = model(observations)

            shadow_distributions[agent] = Categorical(probs_)

        with torch.no_grad():
            kl_divs = {
                f'kl_{self.i}_{i}': kl_divergence(action_distributions[self.agent], shadow_distributions[f'agent_{i}']).mean()
                for i, _ in enumerate(self.models)
            }
            logs.update(kl_divs)

    def after_step(self,
                   logs: Optional[dict],
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        for i, model in enumerate(self.models):
            agent = f'agent_{i}'
            self.hidden_states[agent][(dones['__all__'] & ~self.mask_env_dones) | (dones[self.agent] & ~self.mask_agent_dones)] = 0
            self.cell_states[agent][(dones['__all__'] & ~self.mask_env_dones) | (dones[self.agent] & ~self.mask_agent_dones)] = 0

        self.hidden_states = {k: v.detach() for k, v in self.hidden_states.items()}
        self.cell_states = {k: v.detach() for k, v in self.cell_states.items()}