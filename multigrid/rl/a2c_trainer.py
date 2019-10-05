import torch
from torch import nn, optim
from typing import Optional, Dict, List, Callable
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

from .a2c_loss import ActorCritic
from .core import SingleAgentTrainer
from .trajectory_store import TrajectoryStore
from multigrid.interaction import Interaction


class A2CTrainer(SingleAgentTrainer):
    def __init__(self,
                 agent_type: str,
                 pool_id: int,
                 model: nn.Module,
                 update_steps: int,
                 optimizer: Optimizer,
                 a2c: ActorCritic,
                 max_grad_norm: float,
                 value_loss_coeff: float,
                 entropy_loss_coeff: float,
                 mask_dones: bool = False,
                 value_loss_fn: Callable = F.smooth_l1_loss):
        super(A2CTrainer, self).__init__()
        self.agent_id = agent_type
        self.pool_id = pool_id
        self.update_steps = update_steps
        self.optimizer = optimizer
        self.model = model
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.mask_dones = mask_dones
        self.max_grad_norm = max_grad_norm
        self.a2c = a2c
        self.value_loss_fn = value_loss_fn

        self.trajectories = TrajectoryStore()
        self.i = 0

    def train(self,
              interaction: Interaction,
              hidden_states: Dict[str, torch.Tensor],
              cell_states: Dict[str, torch.Tensor],
              logs: Optional[dict],
              obs: Optional[Dict[str, torch.Tensor]] = None,
              rewards: Optional[Dict[str, torch.Tensor]] = None,
              dones: Optional[Dict[str, torch.Tensor]] = None,
              infos: Optional[Dict[str, torch.Tensor]] = None,
              current_obs: Optional[Dict[str, torch.Tensor]] = None,
              current_hiddens: Optional[Dict[str, torch.Tensor]] = None,
              current_cells: Optional[Dict[str, torch.Tensor]] = None):
        self.trajectories.append(
            action=interaction.actions[self.agent_id].unsqueeze(-1),
            log_prob=interaction.log_probs[self.agent_id].unsqueeze(-1),
            value=interaction.state_values[self.agent_id],
            reward=rewards[self.agent_id].unsqueeze(-1),
            done=dones[self.agent_id].unsqueeze(-1),
            entropy=interaction.action_distributions[self.agent_id].entropy().unsqueeze(-1)
        )

        if self.i % self.update_steps == 0:
            with torch.no_grad():
                _, bootstrap_values, _ = self.model(obs[self.agent_id], hidden_states[self.agent_id])

            returns = self.a2c.returns(
                bootstrap_values.detach(),
                self.trajectories.rewards,
                self.trajectories.values.detach(),
                self.trajectories.log_probs.detach(),
                torch.zeros_like(self.trajectories.dones) if self.mask_dones else self.trajectories.dones,
            )

            value_loss = self.value_loss_fn(self.trajectories.values, returns).mean()
            advantages = returns - self.trajectories.values
            policy_loss = - (advantages.detach() * self.trajectories.log_probs).mean()

            entropy_loss = - self.trajectories.entropies.mean()

            self.optimizer.zero_grad()
            loss = self.value_loss_coeff * value_loss + policy_loss + self.entropy_loss_coeff * entropy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            self.trajectories.clear()
            hidden_states[self.agent_id] = hidden_states[self.agent_id].detach()
            cell_states[self.agent_id] = cell_states[self.agent_id].detach()

        self.i = (self.i + 1) % self.update_steps
