import torch
from torch import nn
from typing import Optional, Dict, List
import os
import boto3

from multigrid.core import Callback


class ModelCheckpoint(Callback):
    """Periodically saves models.

    `filepath` can contain named formatting options, which will be filled with
    the values of keys in `logs` (passed in `after_step`).

    For example: if `filepath` is `steps={steps:.2e}__species={i_species}.pt`
    then the checkpoint will be saved with the number of training steps and
    the species index.

    Args:
        save_folder:
        filepath:
        models:
        interval:
    """
    def __init__(self,
                 save_folder: str,
                 filepath: str,
                 models: List[nn.Module],
                 interval: Optional[int] = 500000,
                 s3_bucket: Optional[str] = None,
                 s3_filepath: Optional[str] = None):
        super(ModelCheckpoint, self).__init__()
        self.save_folder = save_folder
        self.filepath = filepath
        self.interval = interval
        self.models = models
        self.s3_bucket = s3_bucket
        self.s3_filepath = s3_filepath

        os.makedirs(save_folder, exist_ok=True)

        self.last_saved_steps = 0
        self.num_checkpoints = 0

    def after_train(self,
                    logs: Optional[dict],
                    obs: Optional[Dict[str, torch.Tensor]] = None,
                    rewards: Optional[Dict[str, torch.Tensor]] = None,
                    dones: Optional[Dict[str, torch.Tensor]] = None,
                    infos: Optional[Dict[str, torch.Tensor]] = None):
        # if (logs['steps'] - self.last_saved_steps) >= self.interval:
        #     self.last_saved_steps = self.num_checkpoints*self.interval
        #     self.num_checkpoints += 1
        #
        #     for i, model in enumerate(self.models):
        #         filepath = self.filepath.format(i_agent=i, pool_id=model.pool_id, model_steps=model.train_steps,
        #                                         model_episodes=model.train_episodes, **logs)
        #         torch.save(model.state_dict(), f'{self.save_folder}/{filepath}')
        #         if self.s3_bucket is not None:
        #             boto3.client('s3').upload_file(
        #                 f'{self.save_folder}/{filepath}',
        #                 self.s3_bucket,
        #                 self.s3_filepath.format(i_agent=i, pool_id=model.pool_id, model_steps=model.train_steps,
        #                                         model_episodes=model.train_episodes, **logs)
        #             )

        for i, model in enumerate(self.models):
            if (model.train_steps - model.last_saved_steps) >= self.interval:
                model.last_saved_steps = model.num_checkpoints * self.interval
                model.num_checkpoints += 1

                filepath = self.filepath.format(i_agent=i, pool_id=model.pool_id, model_steps=model.train_steps,
                                                model_episodes=model.train_episodes, **logs)
                torch.save(model.state_dict(), f'{self.save_folder}/{filepath}')
                if self.s3_bucket is not None:
                    boto3.client('s3').upload_file(
                        f'{self.save_folder}/{filepath}',
                        self.s3_bucket,
                        self.s3_filepath.format(i_agent=i, pool_id=model.pool_id, model_steps=model.train_steps,
                                                model_episodes=model.train_episodes, **logs)
                    )
