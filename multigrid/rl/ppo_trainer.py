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

                    clipped_values = values_batch + (new_values - values_batch).clamp(
                        -self.eta_clip, self.eta_clip)
                    value_loss = torch.max(self.value_loss_fn(new_values, returns_batch),
                                           self.value_loss_fn(clipped_values, returns_batch)).mean()

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


# class MultiagentPPOTrainer(MultiAgentTrainer):
#     def __init__(self,
#                  models: List[nn.Module],
#                  update_steps: int,
#                  epochs: int,
#                  interaction_handler: InteractionHandler,
#                  lr: float,
#                  batch_size: int,
#                  gamma: float,
#                  gae_lambda: float,
#                  eta_clip: float,
#                  max_grad_norm: float,
#                  value_loss_coeff: float,
#                  entropy_loss_coeff: float,
#                  normalise_advantages: bool = True,
#                  mask_dones: bool = False,
#                  value_loss_fn: Callable = F.smooth_l1_loss,
#                  dtype: torch.dtype = torch.float):
#         super(MultiagentPPOTrainer, self).__init__()
#         self.models = models
#         self.update_steps = update_steps
#         self.epochs = epochs
#         self.interaction_handler = interaction_handler
#         self.lr = lr
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.gae_lambda = gae_lambda
#         self.eta_clip = eta_clip
#         self.value_loss_coeff = value_loss_coeff
#         self.entropy_loss_coeff = entropy_loss_coeff
#         self.mask_dones = mask_dones
#         self.max_grad_norm = max_grad_norm
#         self.normalise_advantages = normalise_advantages
#         self.use_gae = gae_lambda is not None
#         self.value_loss_fn = value_loss_fn
#         self.dtype = dtype
#
#         self.optimizers = []
#         for i, m in enumerate(models):
#             self.optimizers.append(
#                 optim.Adam(m.parameters(), lr=lr, weight_decay=0)
#             )
#         self.trajectories = [TrajectoryStore() for _, _ in enumerate(models)]
#
#         self.a2c = ActorCritic(gamma=gamma, normalise_returns=False, dtype=dtype,
#                                use_gae=gae_lambda is not None, gae_lambda=gae_lambda)
#
#         self.i = 0
#
#     def _generate_batches(self, *tensors: torch.Tensor):
#         num_envs = tensors[0].shape[1]
#
#         sampler = BatchSampler(
#             SubsetRandomSampler(range(num_envs)),
#             self.batch_size,
#             drop_last=True
#         )
#
#         for indices in sampler:
#             batch = []
#
#             for t in tensors:
#                 batch.append(t[:, indices].view(-1, *t.shape[2:]))
#
#             yield batch
#
#     def train(self,
#               interaction: Interaction,
#               hidden_states: Dict[str, torch.Tensor],
#               cell_states: Dict[str, torch.Tensor],
#               logs: Optional[dict],
#               obs: Optional[Dict[str, torch.Tensor]] = None,
#               rewards: Optional[Dict[str, torch.Tensor]] = None,
#               dones: Optional[Dict[str, torch.Tensor]] = None,
#               infos: Optional[Dict[str, torch.Tensor]] = None):
#         for i, _ in enumerate(self.models):
#             self.trajectories[i].append(
#                 state=obs[f'agent_{i}'],
#                 action=interaction.actions[f'agent_{i}'],
#                 log_prob=interaction.log_probs[f'agent_{i}'],
#                 value=interaction.state_values[f'agent_{i}'],
#                 reward=rewards[f'agent_{i}'],
#                 hidden_state=hidden_states[f'agent_{i}'],
#                 done=dones[f'agent_{i}'],
#                 entropy=interaction.action_distributions[f'agent_{i}'].entropy()
#             )
#
#
#         if self.i % self.update_steps == 0 and self.i > 0:
#             with torch.no_grad():
#                 bootstrap_interaction = self.interaction_handler.interact(obs, hidden_states, cell_states)
#
#             returns = [
#                 self._calculate_returns(
#                     bootstrap_interaction.state_values[f'agent_{i}'].squeeze(-1),
#                     self.trajectories[i].rewards,
#                     self.trajectories[i].values.squeeze(-1).detach(),
#                     torch.zeros_like(self.trajectories[i].dones) if self.mask_dones else self.trajectories[i].dones)
#                 for i, _ in enumerate(self.models)
#             ]
#
#             advantages = []
#             for i, _ in enumerate(self.models):
#                 _advantages = returns[i].unsqueeze(-1) - self.trajectories[i].values
#                 if self.normalise_advantages:
#                     _advantages = (_advantages - _advantages.mean()) / (_advantages.std() + 1e-5)
#
#                 advantages.append(_advantages)
#
#             for i, _ in enumerate(self.models):
#                 for epoch in range(self.epochs):
#                     data_generator = self._generate_batches(
#                         self.trajectories[i].states,
#                         self.trajectories[i].hidden_state.detach(),
#                         self.trajectories[i].actions,
#                         self.trajectories[i].values,
#                         returns[i],
#                         self.trajectories[i].log_probs,
#                         advantages[i]
#                     )
#
#                     for batch in data_generator:
#                         obs_batch, hidden_states_batch, actions_batch, values_batch, returns_batch, old_logs_probs_batch, \
#                             advantages_batch = batch
#
#                         action_probabilities, new_values, _ = self.models[i](obs_batch, hidden_states_batch)
#                         new_action_log_probs = Categorical(action_probabilities).log_prob(actions_batch)
#                         new_entropies = Categorical(action_probabilities).entropy()
#
#                         ratio = torch.exp(new_action_log_probs - old_logs_probs_batch.detach())
#                         surr1 = ratio * advantages_batch.detach()
#                         surr2 = torch.clamp(ratio, 1.0 - self.eta_clip, 1.0 + self.eta_clip) * advantages_batch.detach()
#                         policy_loss = -torch.min(surr1, surr2).mean()
#
#                         # value_loss = self.value_loss_fn(new_values.squeeze(-1), returns_batch).mean()
#                         clipped_values = values_batch.detach() + (new_values - values_batch.detach()).clamp(-self.eta_clip, self.eta_clip)
#                         value_loss = torch.max(self.value_loss_fn(new_values.squeeze(-1), returns_batch),
#                                                self.value_loss_fn(clipped_values.squeeze(-1), returns_batch)).mean()
#
#                         entropy_loss = - new_entropies.mean()
#
#                         self.optimizers[i].zero_grad()
#                         loss = self.value_loss_coeff * value_loss + policy_loss + self.entropy_loss_coeff * entropy_loss
#                         loss.backward()
#                         nn.utils.clip_grad_norm_(self.models[i].parameters(), self.max_grad_norm)
#                         self.optimizers[i].step()
#
#             hidden_states = {k: v.detach() for k, v in hidden_states.items()}
#             cell_states = {k: v.detach() for k, v in cell_states.items()}
#             for traj in self.trajectories:
#                 traj.clear()
#
#         self.i += 1
#
#     def _calculate_returns(self,
#                            bootstrap_values: torch.Tensor,
#                            rewards: torch.Tensor,
#                            values: torch.Tensor,
#                            dones: torch.Tensor):
#         returns = []
#         if self.use_gae:
#             gae = torch.zeros_like(bootstrap_values).requires_grad_(False)
#             for t in reversed(range(rewards.size(0))):
#                 if t == rewards.size(0) - 1:
#                     delta = rewards[t] + self.gamma * bootstrap_values * (~dones[t]).to(self.dtype) - values[t]
#                 else:
#                     delta = rewards[t] + self.gamma * values[t + 1] * (~dones[t]).to(self.dtype) - values[t]
#
#                 gae = delta + self.gamma * self.gae_lambda * (~dones[t]).to(self.dtype) * gae
#                 R = gae + values[t]
#                 returns.insert(0, R)
#         else:
#             R = bootstrap_values * (~dones[-1]).to(self.dtype)
#             for r, d in zip(reversed(rewards), reversed(dones)):
#                 R = r + self.gamma * R * (~d).to(self.dtype)
#                 returns.insert(0, R)
#
#         returns = torch.stack(returns)
#
#         return returns
