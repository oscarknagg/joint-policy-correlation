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
    log_path = f'{PATH}/experiments/{args.save_folder}/{args.model_checkpoint_steps}/{i}-{j}.csv'
    # Check for already complete
    if os.path.exists(log_path):
        print('({}, {}), already complete.'.format(i, j))
        return

    env = arguments.get_env(experiment_args, observation_function, experiment_args.device)

    # Generic get-models
    try:
        model_locations = [
            models_1[i],
            models_2[j]
        ]
        print('-'*max(len(model_locations[0]), len(model_locations[1])))
        print(model_locations[0])
        print('...VS...')
        print(model_locations[1])
        print(log_path)
        print('-' * max(len(model_locations[0]), len(model_locations[1])))
    except KeyError as e:
        warnings.warn('Model ({}, {}) not found.'.format(i, j))
        return

    models = []
    for model_path in model_locations:
        models.append(
            agents.GRUAgent(
                num_actions=env.num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=experiment_args.device,
                                                                                dtype=experiment_args.dtype)
        )
        models[-1].load_state_dict(torch.load(model_path))

    interaction_handler = MultiSpeciesHandler(models, experiment_args.n_species, experiment_args.n_agents,
                                              experiment_args.agent_type, keep_obs=False)

    callbacks = [
        loggers.LoggingHandler(env),
        loggers.CSVLogger(
            filename=log_path,
        ),
        loggers.VideoLogger(
            env,
            f'{PATH}/experiments/{args.save_folder}/videos/{args.model_checkpoint_steps}/{i}-{j}/'
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


def get_models(folder, steps, species):
    # print(folder, steps, species)
    models_to_run = {}
    for root, _, files in os.walk(folder):
        for f in files:
            model_args = {i.split('=')[0]: i.split('=')[1] for i in f[:-3].split('__')}
            # print(f)
            if float(model_args['steps']) == steps and int(model_args['species']) == species:
                models_to_run.update({
                    int(model_args['repeat']): os.path.join(root, f)
                })

    return models_to_run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-1', type=str)
    parser.add_argument('--folder-2', type=str)
    parser.add_argument('--species-1', type=int)
    parser.add_argument('--species-2', type=int)
    parser.add_argument('--model-checkpoint-steps', type=float)
    parser = arguments.add_common_arguments(parser)
    parser = arguments.add_training_arguments(parser)
    parser = arguments.add_model_arguments(parser)
    parser = arguments.add_observation_arguments(parser)
    parser = arguments.add_snake_env_arguments(parser)
    parser = arguments.add_laser_tag_env_arguments(parser)
    parser = arguments.add_render_arguments(parser)
    parser = arguments.add_output_arguments(parser)
    args = parser.parse_args()

    # Read args file and get original arguments
    folder_1 = os.path.join('experiments', args.folder_1, 'models')
    folder_2 = os.path.join('experiments', args.folder_2, 'models')

    argsfile = f'{PATH}/experiments/{args.folder_1}/args.txt'
    old_args = open(argsfile).read().replace('python ', '')
    experiment_args = parser.parse_args(old_args.split())
    experiment_args.dtype = arguments.get_dtype(experiment_args.dtype)

    models_1 = get_models(folder_1, args.model_checkpoint_steps, args.species_1)
    models_2 = get_models(folder_2, args.model_checkpoint_steps, args.species_2)

    n = max({k for k in models_1.keys()}) + 1
    m = max({k for k in models_2.keys()}) + 1

    # Configure observations
    observation_function = observations.FirstPersonCrop(
        height=experiment_args.obs_h,
        width=experiment_args.obs_w,
        first_person_rotation=experiment_args.obs_rotate,
        in_front=experiment_args.obs_in_front,
        behind=experiment_args.obs_behind,
        side=experiment_args.obs_side
    )

    indices = list(product(range(n), range(m)))

    pool = Pool(args.n_processes)
    try:
        pool.starmap(worker, indices)
    except KeyboardInterrupt:
        raise Exception
    finally:
        pool.close()
