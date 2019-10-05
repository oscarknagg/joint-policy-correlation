import torch
from typing import Dict, Optional, List

from multigrid.core import Callback


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
                 diversity_coeff: float,
                 retrain_interval: int,
                 window_length: int,
                 experiment_folder: str,
                 matchup: List[int]):
        super(DiversityReward, self).__init__()
        self.diversity_coeff = diversity_coeff
        self.retrain_interval = retrain_interval
        self.window_length = window_length
        self.experiment_folder = experiment_folder
        self.matchup = matchup
        self.i = 0

    def _retrain(self):
        pass

    def _diversity_reward(self, obs, rewards):
        # Run network on observations to get predictions
        # Use ground truth IDs to calculate loss
        # Subtract loss * diversity_coeff from rewards
        pass

    def after_step(self,
                   logs: Optional[dict],
                   obs: Optional[Dict[str, torch.Tensor]] = None,
                   rewards: Optional[Dict[str, torch.Tensor]] = None,
                   dones: Optional[Dict[str, torch.Tensor]] = None,
                   infos: Optional[Dict[str, torch.Tensor]] = None):
        # Save logs to file here
        pass

        if self.i % self.retrain_interval:
            self._retrain()

        self._diversity_reward(obs, rewards)

        # Add intrinsic reward to logs

    def on_train_end(self):
        # Clean up trajectories folder
        pass
