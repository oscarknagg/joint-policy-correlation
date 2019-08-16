"""Takes a folder full of agents and calculates JPC."""
import os
import argparse
from itertools import product
from multiprocessing import Pool
import torch
import warnings

from multigrid.core import MultiAgentRun
from multigrid.core import CallbackList
from multigrid.callbacks import loggers
from multigrid.interaction import MultiSpeciesHandler
from multigrid import agents
from multigrid import observations
from multigrid import arguments
from config import PATH, INPUT_CHANNELS


def worker(i, j):
    # Check for already complete
    if os.path.exists(f'{PATH}/experiments/{args.experiment_folder}/jpc/{args.model_checkpoint_steps}/{i}-{j}.csv'):
        print('({}, {}), already complete.'.format(i, j))
        return

    env = arguments.get_env(experiment_args, observation_function, args.device)

    # Generic get-models
    try:
        model_locations = [
            models_to_run[i, 0],
            models_to_run[j, 1]
        ]
    except KeyError as e:
        warnings.warn('Model ({}, {}) not found.'.format(i, j))
        return

    models = []
    for model_path in model_locations:
        models.append(
            agents.RecurrentAgent(recurrent_module=args.agent_type, num_actions=env.num_actions,
                                  in_channels=INPUT_CHANNELS, channels=16, fc_size=32, hidden_size=32
                                  ).to(device=args.device, dtype=args.dtype)
        )
        models[-1].load_state_dict(torch.load(model_path, map_location=args.device))

    interaction_handler = MultiSpeciesHandler(models, experiment_args.n_species, experiment_args.n_agents, experiment_args.agent_type, keep_obs=False)
    print(f'{PATH}/experiments/{args.experiment_folder}/jpc/{i}-{j}.csv')
    callbacks = [
        loggers.LogEnricher(env),
        loggers.CSVLogger(
            filename=f'{PATH}/experiments/{args.experiment_folder}/jpc/{args.model_checkpoint_steps}/{i}-{j}.csv',
        ),
        loggers.VideoLogger(
            env,
            f'{PATH}/experiments/{args.experiment_folder}/videos/jpc/{args.model_checkpoint_steps}/{i}-{j}/'
        ) if args.save_video else None,

    ]
    callbacks = [c for c in callbacks if c]
    callback_list = CallbackList(callbacks)
    callback_list.on_train_begin()

    environment_run = MultiAgentRun(
        env=env,
        models=models,
        interaction_handler=interaction_handler,
        callbacks=callback_list,
        trainers=None,
        total_episodes=experiment_args.n_envs,
    )
    environment_run.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-folder', type=str)
    parser.add_argument('--model-checkpoint-steps', type=float)
    parser = arguments.add_common_arguments(parser)
    parser = arguments.add_training_arguments(parser)
    parser = arguments.add_model_arguments(parser)
    parser = arguments.add_observation_arguments(parser)
    parser = arguments.add_snake_env_arguments(parser)
    parser = arguments.add_laser_tag_env_arguments(parser)
    parser = arguments.add_treasure_hunt_env_arguments(parser)
    parser = arguments.add_render_arguments(parser)
    parser = arguments.add_output_arguments(parser)
    args = parser.parse_args()

    # Read args file and get original arguments
    model_folder = os.path.join('experiments', args.experiment_folder, 'models')
    argsfile = f'{PATH}/experiments/{args.experiment_folder}/args.txt'
    old_args = open(argsfile).read().replace('python ', '')
    experiment_args = parser.parse_args(old_args.split())
    experiment_args.dtype = arguments.get_dtype(experiment_args.dtype)
    args.dtype = arguments.get_dtype(args.dtype)

    # Find all models to run by searching the models folder for all species and repeats of
    # a certain number of training steps
    models_to_run = {}
    for root, _, files in os.walk(model_folder):
        for f in files:
            model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
            if float(model_args['steps']) == args.model_checkpoint_steps:
                models_to_run.update({
                    (int(model_args['repeat']), int(model_args['species'])): os.path.join(root, f)
                })

    n_repeats = max({k[0] for k in models_to_run.keys()}) + 1
    n_species = max({k[1] for k in models_to_run.keys()}) + 1

    # Configure observations
    observation_function = observations.FirstPersonCrop(
        height=experiment_args.obs_h,
        width=experiment_args.obs_w,
        first_person_rotation=experiment_args.obs_rotate,
        in_front=experiment_args.obs_in_front,
        behind=experiment_args.obs_behind,
        side=experiment_args.obs_side
    )

    indices = list(product(range(n_repeats), range(n_repeats)))

    pool = Pool(args.n_processes)
    try:
        pool.starmap(worker, indices)
    except KeyboardInterrupt:
        raise Exception
    finally:
        pool.close()
