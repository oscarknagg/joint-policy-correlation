import torch


class TrajectoryStore(object):
    """Stores list of transitions.

    Each property should return a tensor of shape (num_steps, num_envs, 1)
    """
    def __init__(self):
        self.clear()

    def append(self,
               obs: torch.Tensor = None,
               action: torch.Tensor = None,
               log_prob: torch.Tensor = None,
               reward: torch.Tensor = None,
               value: torch.Tensor = None,
               done: torch.Tensor = None,
               entropy: torch.Tensor = None,
               hidden_state: torch.Tensor = None,
               cell_state: torch.Tensor = None):
        """Adds a transition to the store.

        Each argument should be a vector of shape (num_envs, 1)
        """
        if obs is not None:
            self._states.append(obs)

        if action is not None:
            self._actions.append(action)

        if log_prob is not None:
            self._log_probs.append(log_prob)

        if reward is not None:
            self._rewards.append(reward)

        if value is not None:
            self._values.append(value)

        if done is not None:
            self._dones.append(done)

        if entropy is not None:
            self._entropies.append(entropy)

        if hidden_state is not None:
            self._hiddens.append(hidden_state)

        if cell_state is not None:
            self._cells.append(hidden_state)

    def clear(self):
        self._states = []
        self._actions = []
        self._log_probs = []
        self._rewards = []
        self._values = []
        self._dones = []
        self._entropies = []
        self._hiddens = []
        self._cells = []

    @property
    def obs(self):
        return torch.stack(self._states)

    @property
    def actions(self):
        return torch.stack(self._actions)

    @property
    def log_probs(self):
        return torch.stack(self._log_probs)

    @property
    def rewards(self):
        return torch.stack(self._rewards)

    @property
    def values(self):
        return torch.stack(self._values)

    @property
    def dones(self):
        return torch.stack(self._dones)

    @property
    def entropies(self):
        return torch.stack(self._entropies)

    @property
    def hidden_state(self):
        return torch.stack(self._hiddens)

    @property
    def cell_state(self):
        return torch.stack(self._hiddens)
