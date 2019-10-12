import torch
from torch import nn
from typing import Optional, List, Dict
from itertools import product
from time import time
import math
import argparse

from multigrid.core import MultiAgentRun, MultiagentVecEnv, MultiAgentTrainer, InteractionHandler, CallbackList
from multigrid.interaction import AgentPoolHandler, MultiSpeciesHandler
from multigrid.rl.core import IndependentTrainer
from multigrid.callbacks import loggers
from multigrid import callbacks
from multigrid import utils
from config import PATH


class MultiAgentPoolRun(object):
    def __init__(self, env: MultiagentVecEnv,
                 n_pool: int,
                 models: Dict[str, List[nn.Module]],
                 trainers: Optional[MultiAgentTrainer],
                 mask_dones: bool = False,
                 warm_start: int = 0,
                 initial_steps: int = 0,
                 initial_episodes: int = 0,
                 schedule_steps: int = 1000,
                 repeat_number: int = 0,
                 total_steps: Optional[int] = None,
                 total_episodes: Optional[int] = None,
                 args: argparse.Namespace = None):
        self.env = env
        self.n_pool = n_pool
        self.models = models
        self.trainers = trainers
        self.train = trainers is not None

        self.mask_dones = mask_dones
        self.warm_start = warm_start
        self.initial_steps = initial_steps
        self.initial_episodes = initial_episodes
        self.schedule_steps = schedule_steps
        self.repeat_number = repeat_number
        self.total_steps = total_steps if total_steps else float('inf')
        self.total_episodes = total_episodes if total_episodes else float('inf')
        self.args = args

        n = math.ceil(self.total_steps / (schedule_steps * self.env.num_envs))

        # Random schedule
        self.schedule = []
        for i in range(self.env.num_agents):
            schedule = torch.cat([torch.ones(n) * i for i in range(self.n_pool)])
            schedule = schedule[torch.randperm(schedule.size(0))]
            self.schedule.append(schedule)

        self.schedule = torch.stack(self.schedule).t()

    def run(self):
        train_begin_time = time()
        total_steps_per_agent = {
            f'agent_{i_agent_type}': [0, ] * self.n_pool for i_agent_type in range(self.env.num_agents)
        }
        total_episodes_per_agent = {
            f'agent_{i_agent_type}': [0, ] * self.n_pool for i_agent_type in range(self.env.num_agents)
        }
        total_steps = 0
        total_episodes = 0

        # Define long-lived callbacks
        diversity_callback = callbacks.DiversityReward(
            diversity_coeff=self.args.diversity,
            retrain_interval=1000,
            window_length=2000,
            experiment_folder=f'{PATH}/experiments/{self.args.save_folder}',
            matchup=[],
            num_pool=self.n_pool
        ) if self.args.diversity != 0 else None

        for i_matchup, matchup in enumerate(self.schedule):
            models_to_train = {
                f'agent_{i_agent_type}_{int(i_model.item())}': self.models[f'agent_{i_agent_type}'][int(i_model.item())]
                for i_agent_type, i_model in enumerate(matchup)
            }
            model_trainers = [
                self.trainers[f'agent_{i_agent_type}'][int(i_model.item())]
                for i_agent_type, i_model in enumerate(matchup)
            ]
            rl_trainer = IndependentTrainer(model_trainers)
            interaction_handler = MultiSpeciesHandler(list(models_to_train.values()), n_agents=self.env.num_agents, keep_obs=True)

            save_file = f'repeat={self.repeat_number}'
            model_save_format_string = 'steps={model_steps:.2e}__agent={i_agent}__pool_id={pool_id}.pt'

            if self.args.diversity != 0:
                diversity_callback.matchup = list(matchup)

            callback_list = [
                loggers.LoggingHandler(
                    self.env,
                    initial_time=train_begin_time,
                    initial_total_steps=total_steps,
                    initial_total_episodes=total_episodes,
                    agent_ids=matchup.int().tolist(),
                    agent_steps=[total_steps_per_agent[f'agent_{i_agent_type}'][int(i_model.item())]
                                 for i_agent_type, i_model in enumerate(matchup)],
                    agent_episodes=[total_episodes_per_agent[f'agent_{i_agent_type}'][int(i_model.item())]
                                    for i_agent_type, i_model in enumerate(matchup)]
                ),
                loggers.PrintLogger(
                    env=self.env,
                    interval=self.args.print_interval
                ) if self.args.print_interval is not None else None,
                callbacks.ModelCheckpoint(
                    f'{PATH}/experiments/{self.args.save_folder}/models',
                    f'repeat={self.repeat_number}__' + model_save_format_string,
                    list(models_to_train.values()),
                    interval=self.args.model_interval,
                    s3_bucket=self.args.s3_bucket,
                    s3_filepath=f'{self.args.save_folder}/models/repeat={self.repeat_number}__' + model_save_format_string
                ) if self.args.save_model else None,
                diversity_callback,
                loggers.CSVLogger(
                    filename=f'{PATH}/experiments/{self.args.save_folder}/logs/{save_file}.csv',
                    header_comment=utils.get_comment(self.args),
                    interval=self.args.log_interval,
                    append=i_matchup > 0,
                    # append=resume_data.resume or i_matchup == 0,
                    s3_bucket=self.args.s3_bucket,
                    s3_filename=f'{self.args.save_folder}/logs/{save_file}.csv',
                    s3_interval=self.args.s3_interval
                ) if self.args.save_logs else None,
            ]
            callback_list = [c for c in callback_list if c]
            callback_list = CallbackList(callback_list)
            callback_list.on_train_begin()

            multiagent_run = MultiAgentRun(
                env=self.env,
                models=list(models_to_train.values()),
                interaction_handler=interaction_handler,
                callbacks=callback_list,
                trainers=rl_trainer,
                mask_agent_dones=self.mask_dones,
                warm_start=0,
                initial_steps=total_steps,
                total_steps=total_steps + self.schedule_steps * self.env.num_envs,
                initial_episodes=total_episodes,
            )
            _, logs = multiagent_run.run()
            final_logs = logs[-1]

            total_steps = final_logs['steps']
            total_episodes = final_logs['episodes']
            for i_agent_type, i_model in enumerate(matchup):
                total_steps_per_agent[f'agent_{i_agent_type}'][int(i_model.item())] = final_logs[f'agent_steps_{i_agent_type}']
                total_episodes_per_agent[f'agent_{i_agent_type}'][int(i_model.item())] = final_logs[f'agent_episodes_{i_agent_type}']

            print('Total steps ', final_logs['steps'])
            print('Total episodes ', final_logs['episodes'])
            print('Steps per agent', total_steps_per_agent)
            print('Episodes per agent', total_episodes_per_agent)
