import torch
from torch import nn, optim
from typing import List, Optional, Dict, Callable
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from torch.optim.optimizer import Optimizer

from .core import MultiAgentTrainer, SingleAgentTrainer
from .a2c_loss import ActorCritic
from .trajectory_store import TrajectoryStore
from multigrid.interaction import Interaction, InteractionHandler
from multigrid import utils

class PPOTrainer(SingleAgentTrainer):
    def __init__(self,
                 agent_id: str,
                 model: nn.Module,
                 update_steps: int,
                 optimizer: Optimizer,
                 a2c: ActorCritic,
                 epochs: int,
                 batch_size: int,
                 eta_clip: float,
                 max_grad_norm: float,
                 value_loss_coeff: float,
                 entropy_loss_coeff: float,
                 normalise_advantages: bool = True,
                 mask_dones: bool = False,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 dtype: torch.dtype = torch.float):
        self.agent_id = agent_id
        self.model = model
        self.update_steps = update_steps
        self.a2c = a2c
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta_clip = eta_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.mask_dones = mask_dones
        self.max_grad_norm = max_grad_norm
        self.normalise_advantages = normalise_advantages
        self.value_loss_fn = value_loss_fn
        self.dtype = dtype

        self.trajectories = TrajectoryStore()
        self.i = 0

    def _generate_batches(self, *tensors: torch.Tensor):
        num_envs = tensors[0].shape[1]

        sampler = BatchSampler(
            SubsetRandomSampler(range(num_envs)),
            self.batch_size,
            drop_last=True
        )

        for indices in sampler:
            batch = []

            for t in tensors:
                batch.append(t[:, indices].view(-1, *t.shape[2:]))

            yield batch

    def train(self,
              interaction: Interaction,
              hidden_states: Dict[str, torch.Tensor],
              cell_states: Dict[str, torch.Tensor],
              logs: Optional[dict],
              obs: Optional[Dict[str, torch.Tensor]] = None,
              rewards: Optional[Dict[str, torch.Tensor]] = None,
              dones: Optional[Dict[str, torch.Tensor]] = None,
              infos: Optional[Dict[str, torch.Tensor]] = None):
        self.trajectories.append(
            obs=obs[self.agent_id],
            hidden_state=hidden_states[self.agent_id],
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

            advantages = returns - self.trajectories.values

            for epoch in range(self.epochs):
                data_generator = self._generate_batches(
                    self.trajectories.obs,
                    self.trajectories.hidden_state.detach(),
                    self.trajectories.actions.detach(),
                    self.trajectories.values.detach(),
                    # self.trajectories.values,
                    returns,
                    self.trajectories.log_probs.detach(),
                    advantages
                )
                for batch in data_generator:
                    obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, old_logs_probs_batch, \
                        advantages_batch = batch

                    # value_loss = self.value_loss_fn(values_batch, returns_batch).mean()
                    # advantages = returns - self.trajectories.values
                    # policy_loss = - (advantages.detach() * self.trajectories.log_probs).mean()
                    # entropy_loss = - self.trajectories.entropies.mean()

                    # Vanilla A2C
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


class _PPOTrainer(SingleAgentTrainer):
    def __init__(self,
                 agent_id: str,
                 model: nn.Module,
                 update_steps: int,
                 optimizer: Optimizer,
                 a2c: ActorCritic,
                 epochs: int,
                 batch_size: int,
                 eta_clip: float,
                 max_grad_norm: float,
                 value_loss_coeff: float,
                 entropy_loss_coeff: float,
                 normalise_advantages: bool = True,
                 mask_dones: bool = False,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 dtype: torch.dtype = torch.float):
        self.agent_id = agent_id
        self.model = model
        self.update_steps = update_steps
        self.a2c = a2c
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.eta_clip = eta_clip
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.mask_dones = mask_dones
        self.max_grad_norm = max_grad_norm
        self.normalise_advantages = normalise_advantages
        self.value_loss_fn = value_loss_fn
        self.dtype = dtype

        self.trajectories = TrajectoryStore()
        self.i = 0

    def _generate_batches(self, *tensors: torch.Tensor):
        num_envs = tensors[0].shape[1]

        sampler = BatchSampler(
            SubsetRandomSampler(range(num_envs)),
            self.batch_size,
            drop_last=True
        )

        for indices in sampler:
            batch = []

            for t in tensors:
                batch.append(t[:, indices].view(-1, *t.shape[2:]))

            yield batch

    def train(self,
              interaction: Interaction,
              hidden_states: Dict[str, torch.Tensor],
              cell_states: Dict[str, torch.Tensor],
              logs: Optional[dict],
              obs: Optional[Dict[str, torch.Tensor]] = None,
              rewards: Optional[Dict[str, torch.Tensor]] = None,
              dones: Optional[Dict[str, torch.Tensor]] = None,
              infos: Optional[Dict[str, torch.Tensor]] = None):
        self.trajectories.append(
            obs=obs[self.agent_id],
            hidden_state=hidden_states[self.agent_id],
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

            advantages = returns - self.trajectories.values

            for epoch in range(self.epochs):
                data_generator = self._generate_batches(
                    self.trajectories.obs,
                    self.trajectories.hidden_state.detach(),
                    self.trajectories.actions.detach(),
                    self.trajectories.values.detach(),
                    returns,
                    self.trajectories.log_probs.detach(),
                    advantages
                )

                for batch in data_generator:
                    obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, old_logs_probs_batch, \
                        advantages_batch = batch

                    action_probabilities, new_values, _ = self.model(obs_batch, hidden_states_batch)
                    new_action_log_probs = Categorical(action_probabilities).log_prob(actions_batch)
                    new_entropies = Categorical(action_probabilities).entropy()

                    ratio = torch.exp(new_action_log_probs - old_logs_probs_batch.detach())
                    surr1 = ratio * advantages_batch.detach()
                    surr2 = torch.clamp(ratio, 1.0 - self.eta_clip, 1.0 + self.eta_clip) * advantages_batch.detach()
                    policy_loss = -torch.min(surr1, surr2).mean()
                    # No trust region
                    # policy_loss = - (new_action_log_probs * advantages_batch.detach()).mean()

                    clipped_values = values_batch + (new_values - values_batch).clamp(
                        -self.eta_clip, self.eta_clip)
                    value_loss = torch.max(self.value_loss_fn(new_values, returns_batch),
                                           self.value_loss_fn(clipped_values, returns_batch)).mean()
                    # # No value loss clipping
                    # value_loss = self.value_loss_fn(new_values, returns_batch)

                    entropy_loss = - new_entropies.mean()

                    self.optimizer.zero_grad()
                    loss = self.value_loss_coeff * value_loss + policy_loss + self.entropy_loss_coeff * entropy_loss
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            self.trajectories.clear()
            hidden_states[self.agent_id] = hidden_states[self.agent_id].detach()
            cell_states[self.agent_id] = cell_states[self.agent_id].detach()

        self.i = (self.i + 1) % self.update_steps
