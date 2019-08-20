import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F


class RecurrentAgent(nn.Module):
    """Laser tag agent from https://arxiv.org/pdf/1711.00832.pdf.

    A few differences:
    1. Regular conv-ReLU layers instead of concatenated ReLU layers
    2. Remove extra head as this is speacific to the Reactor architecture
    """
    def __init__(self, recurrent_module: str,
                 num_actions: int,
                 in_channels: int,
                 channels: int,
                 fc_size: int,
                 hidden_size: int,
                 batch_norm: bool = False,
                 flattened_size: int = 144):
        super(RecurrentAgent, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.linear = nn.Linear(flattened_size, fc_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.batch_norm = batch_norm

        if self.batch_norm:
            self.bn_conv1 = nn.BatchNorm2d(channels)
            self.bn_conv2 = nn.BatchNorm2d(channels)
            self.bn_conv3 = nn.BatchNorm2d(channels)
            self.bn_linear = nn.BatchNorm1d(hidden_size)

        self.recurrent = recurrent_module
        if self.recurrent == 'gru':
            self.lstm = nn.GRUCell(fc_size, hidden_size)
            self.hidden_size = hidden_size
            self.cell_size = 0
        elif self.recurrent == 'lstm':
            self.lstm = nn.LSTMCell(fc_size, hidden_size)
            self.hidden_size = hidden_size
            self.cell_size = hidden_size

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None, c: Optional[torch.Tensor] = None):
        x = self.conv1(x)
        if self.batch_norm:
            x = F.relu(self.bn_conv1(x))
        else:
            x = F.relu(x)

        x = self.conv2(x)
        if self.batch_norm:
            x = F.relu(self.bn_conv2(x))
        else:
            x = F.relu(x)

        x = self.conv3(x)
        if self.batch_norm:
            x = F.relu(self.bn_conv3(x))
        else:
            x = F.relu(x)

        x = self.linear(x.view(x.size(0), -1))
        if self.batch_norm:
            x = F.relu(self.bn_linear(x))
        else:
            x = F.relu(x)

        if self.recurrent == 'gru':
            h = self.lstm(x, h)
            values = self.value_head(h)
            action_probabilities = self.policy_head(h)
            return F.softmax(action_probabilities, dim=-1), values, h
        elif self.recurrent == 'lstm':
            h, c = self.lstm(x, (h, c))

            values = self.value_head(h)
            action_probabilities = self.policy_head(h)

            return F.softmax(action_probabilities, dim=-1), values, h, c
