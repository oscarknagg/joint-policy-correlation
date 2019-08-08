import torch
from torch import nn
import torch.nn.functional as F


class LSTMAgent(nn.Module):
    """Laser tag agent from https://arxiv.org/pdf/1711.00832.pdf.

    A few differences:
    1. Regular conv-ReLU layers instead of concatenated ReLU layers
    2. Remove extra head as this is speacific to the Reactor architecture
    """
    def __init__(self, num_actions: int,
                 in_channels: int,
                 channels: int,
                 fc_size: int,
                 lstm_size: int,
                 flattened_size: int = 144):
        super(LSTMAgent, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        self.linear = nn.Linear(flattened_size, fc_size)
        self.lstm = nn.LSTMCell(fc_size, lstm_size)
        self.value_head = nn.Linear(lstm_size, 1)
        self.policy_head = nn.Linear(lstm_size, num_actions)

        self.hidden_size = lstm_size
        self.cell_size = lstm_size

    def forward(self, x, h, c):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.linear(x.view(x.size(0), -1)))
        h, c = self.lstm(x, (h, c))

        values = self.value_head(h)
        action_probabilities = self.policy_head(h)

        return F.softmax(action_probabilities, dim=-1), values, h, c
