import torch
from torch import nn, optim
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F
from typing import Dict, Optional, List
import os
import shutil
from tqdm import tqdm
import numpy as np

from multigrid.core import Callback, MultiagentVecEnv
from multigrid.rl import TrajectoryStore


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 fc_size: int,
                 hidden_size: int,
                 n_classes: int,
                 flattened_size: int = 144):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.linear = nn.Linear(flattened_size, fc_size)
        self.n_classes = n_classes
        self.discriminator = nn.Linear(hidden_size, n_classes)

        self.bn_conv1 = nn.BatchNorm2d(channels)
        self.bn_conv2 = nn.BatchNorm2d(channels)
        self.bn_conv3 = nn.BatchNorm2d(channels)
        self.bn_linear = nn.BatchNorm1d(hidden_size)

        self.recurrent = nn.GRU(fc_size, hidden_size)
        self.hidden_size = hidden_size
        self.fc_size = fc_size
        self.cell_size = 0
        for name, param in self.recurrent.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x: torch.Tensor, hx=None, eval=False):
        # if eval:
        #     import pdb; pdb.set_trace()

        seq_len, batch, c, h, w = x.shape
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))

        x = self.conv1(x)
        x = F.relu(self.bn_conv1(x))

        x = self.conv2(x)
        x = F.relu(self.bn_conv2(x))

        x = self.conv3(x)
        x = F.relu(self.bn_conv3(x))

        x = self.linear(x.view(x.size(0), -1))
        x = F.relu(self.bn_linear(x))

        x = x.view(seq_len, batch, self.fc_size)

        x, hx_new = self.recurrent(x, hx)

        x = self.discriminator(x.view(x.size(0) * x.size(1), x.size(2)))
        x = x.view(seq_len, batch, self.n_classes)
        return x, hx_new


def loss_fn(y_pred: torch.Tensor, y_true: int, reduction='mean') -> torch.Tensor:
    seq_len, batch, n_classes = y_pred.shape
    y_true = torch.ones(y_pred.shape[:2], device=y_pred.device, dtype=torch.long) * y_true

    y_pred = y_pred.view(seq_len * batch, n_classes)
    y_true = y_true.view(seq_len * batch)

    loss = F.cross_entropy(y_pred, y_true, reduction=reduction)
    return loss


class DiversityReward(Callback):
    """Applies a reward for agents acting in a distinguishable way within their pool.

    Args:
        diversity_coeff:
        retrain_interval:
        window_length:
        experiment_folder:
        matchup: Ground truth ids of agents within pool [agent_id_0, agent_id_1, etc...]
    """
    def __init__(self,
                 models: List[nn.Module],
                 env: MultiagentVecEnv,
                 diversity_coeff: float,
                 retrain_interval: int,
                 window_length: int,
                 experiment_folder: str,
                 matchup: List[int],
                 num_pool: int,
                 seq_len: int = 125,  # Convenient number: 1/8th of episode length
                 cleanup: bool = False):
        super(DiversityReward, self).__init__()
        self.models = models
        self.env = env
        self.diversity_coeff = diversity_coeff
        self.retrain_interval = retrain_interval
        self.window_length = window_length
        self.experiment_folder = experiment_folder
        self.num_pool = num_pool
        self.matchup = matchup
        self.seq_len = seq_len
        self.cleanup = cleanup
        self.i = 0

        # Discriminator structure hardcoded as I don't see a need to tune it excessively
        self.in_channels = 3
        self.channels = 16
        self.fc_size = 16
        self.hidden_size = 16

        # 2 agent types hardcoded
        self.num_agent_types = 2

        self.epochs = 3

        self.discriminators = [
            Discriminator(
                self.in_channels, self.channels, self.fc_size, self.hidden_size, n_classes=self.num_pool
            ).cuda()
            for _ in range(self.num_agent_types)
        ]
        self.trajectories = [TrajectoryStore() for _ in range(self.num_agent_types)]
        self._reset_hidden_states()

        # Setup directories
        os.makedirs(f'{experiment_folder}/trajectories/', exist_ok=True)
        for agent_type_id in range(self.num_agent_types):
            os.makedirs(f'{experiment_folder}/trajectories/agent_{agent_type_id}', exist_ok=True)
            for pool_id in range(self.num_pool):
                os.makedirs(f'{experiment_folder}/trajectories/agent_{agent_type_id}/{pool_id}/', exist_ok=True)

    def _reset_hidden_states(self):
        self.hidden_states = {
            f'agent_{i}': torch.zeros((self.env.num_envs, self.hidden_size), device=self.env.device)
            for i in range(self.num_agent_types)
        }

    def on_train_begin(self):
        self._reset_hidden_states()

    def _retrain(self):
        # 2 agent types hardcoded
        print('Retraining at step {}'.format(self.i))
        training_logs = {}
        for agent_type_id in range(self.num_agent_types):
            logs = self._fit(agent_type_id)

            for k, v in logs.items():
                training_logs.update({
                    f'discrim_train_loss_{agent_type_id}_{k}': v
                })

        return training_logs

    def _fit(self, agent_type_id: int):
        trajectories = DatasetFolder(
            f'{self.experiment_folder}/trajectories/agent_{agent_type_id}/', extensions=['.pt'], loader=torch.load)
        discriminator = Discriminator(
            self.in_channels, self.channels, self.fc_size, self.hidden_size, n_classes=self.num_pool).cuda()
        discriminator.train()
        optimiser = optim.Adam(discriminator.parameters())

        print('Training discriminator for agent_type {} with {} samples'.format(agent_type_id, len(trajectories)))
        training_logs = {}
        for epoch in range(self.epochs):
            mean_loss = 0
            for x, y in trajectories:
                optimiser.zero_grad()
                y_pred, _ = discriminator(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimiser.step()

                mean_loss += loss.item() / len(trajectories)

            training_logs.update({f'epoch_{epoch}': mean_loss})
            print('Epoch {}: {:.2f}'.format(epoch, mean_loss))

        self.discriminators[agent_type_id] = discriminator

        return training_logs

    def _after_retrain(self):
        # Clean up trajectories so that only the last window length exists
        print('Cleaning up old trajectories')
        for agent_type_id in range(self.num_agent_types):
            pool_id = int(self.matchup[agent_type_id])
            steps = self.models[agent_type_id].train_steps
            print('Agent {} current steps = {}'.format(agent_type_id, steps))
            # Delete anything where t < t_now - self.window_length
            directory = f'{self.experiment_folder}/trajectories/agent_{agent_type_id}/{pool_id}/'
            removed = []
            for root, _, files in os.walk(directory):
                for f in files:
                    j = int(f[:-3])
                    if j <= steps - self.window_length:
                        removed.append(j)
                        os.remove(os.path.join(root, f))

                    # if j < self.i - self.window_length:
                    #     os.remove(os.path.join(root, f))

            print('Removed {}'.format(removed))

    def _diversity_reward(self, obs, rewards):
        # Run network on observations to get predictions
        # Use ground truth IDs to calculate loss
        # Subtract loss * diversity_coeff from rewards
        rewards_dict = {}
        for i, discrim in enumerate(self.discriminators):
            discrim.eval()
            observations = obs[f'agent_{i}'].unsqueeze(0)

            # Predictions of which agent within the pool is being observed
            with torch.no_grad():
                predictions, h_new = discrim(observations, self.hidden_states[f'agent_{i}'].unsqueeze(0), eval=True)
                self.hidden_states[f'agent_{i}'] = h_new.squeeze(0)
                diversity_loss = loss_fn(predictions, self.matchup[i], reduction='none')

            rewards_dict.update({f'diversity_reward_{i}': diversity_loss})

        return rewards_dict

    def _save_observations(self, obs):
        for i, (agent, observations) in enumerate(obs.items()):
            pool_id = int(self.matchup[i])
            trajectories = self.trajectories[i]
            trajectories.append(obs=observations)

            if self.i % self.seq_len == 0:
                steps = self.models[i].train_steps
                torch.save(trajectories.obs, f'{self.experiment_folder}/trajectories/{agent}/{pool_id}/{steps}.pt')
                # torch.save(trajectories.obs, f'{self.experiment_folder}/trajectories/{agent}/{pool_id}/{self.i}.pt')
                trajectories.clear()

    def after_train(self,
                    logs: Optional[dict],
                    obs: Optional[Dict[str, torch.Tensor]] = None,
                    rewards: Optional[Dict[str, torch.Tensor]] = None,
                    dones: Optional[Dict[str, torch.Tensor]] = None,
                    infos: Optional[Dict[str, torch.Tensor]] = None):
        self.i += 1
        self._save_observations(obs)

        if self.i % self.retrain_interval == 0:
            retrain_logs = self._retrain()
            logs.update(retrain_logs)
            self._after_retrain()
        else:
            # Placeholder logs so the CSVLogger will record training losses
            for agent_type_id in range(self.num_agent_types):
                for i in range(self.epochs):
                    logs.update({f'discrim_train_loss_{agent_type_id}_epoch_{i}': np.nan})

        diversity_rewards = self._diversity_reward(obs, rewards)
        logs.update({k: v.mean().item() for k, v in diversity_rewards.items()})

    def on_train_end(self):
        # Clean up trajectories folder
        if self.cleanup:
            shutil.rmtree(f'{self.experiment_folder}/trajectories/', ignore_errors=True)
