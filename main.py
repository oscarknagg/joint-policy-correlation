"""Entry point for training, transferring and visualising agents."""
import argparse
import math
import sys
import os

import torch
from torch import multiprocessing

import multigrid.core
from multigrid import callbacks
from multigrid import core
from multigrid import arguments
from multigrid import utils
from multigrid import resume
from multigrid import observations
from multigrid.callbacks import loggers
from multigrid.interaction import MultiSpeciesHandler
from config import PATH


def worker(repeat: int, device: str, args: argparse.Namespace):
    print('='*20)
    print('Running repeat {} on device {}'.format(repeat, device))
    save_file = f'repeat={repeat}'

    # Resume from checkpoint
    if args.resume_mode == 'local':
        resume_data = resume.LocalResume().resume(args, repeat)
    elif args.resume_mode == 's3':
        resume_data = resume.S3Resume().resume(args, repeat)
    else:
        raise ValueError('Unrecognised resume-mode.')

    if resume_data.current_steps > args.total_steps or resume_data.current_episodes >= args.total_episodes:
        print('Repeat already complete.')
        return

    args.agent_location = resume_data.model_paths

    # Configure env + agents
    observation_function = observations.FirstPersonCrop(
        height=args.obs_h,
        width=args.obs_w,
        first_person_rotation=args.obs_rotate,
        in_front=args.obs_in_front,
        behind=args.obs_behind,
        side=args.obs_side
    )
    env = arguments.get_env(args, observation_function, device)
    models = arguments.get_models(args, env.num_actions, device)
    interaction_handler = MultiSpeciesHandler(models, args.n_species, args.n_agents, args.agent_type,
                                              keep_obs=True if args.train_algo == 'ppo' else False)
    rl_trainer = arguments.get_trainers(args, models)

    callbacks_to_run = [
        loggers.LogEnricher(env, resume_data.current_steps, resume_data.current_episodes),
        loggers.PrintLogger(env=args.env, interval=args.print_interval) if args.print_interval is not None else None,
        callbacks.Render(env, args.fps) if args.render else None,
        callbacks.ModelCheckpoint(
            f'{PATH}/experiments/{args.save_folder}/models',
            f'repeat={repeat}__' + 'steps={steps:.2e}__species={i_species}.pt',
            models,
            interval=args.model_interval,
            s3_bucket=args.s3_bucket,
            s3_filepath=f'{args.save_folder}/models/repeat={repeat}__' + 'steps={steps:.2e}__species={i_species}.pt'
        ) if args.save_model else None,
        loggers.CSVLogger(
            filename=f'{PATH}/experiments/{args.save_folder}/logs/{save_file}.csv',
            header_comment=utils.get_comment(args),
            interval=args.log_interval,
            append=resume_data.resume,
            s3_bucket=args.s3_bucket,
            s3_filename=f'{args.save_folder}/logs/{save_file}.csv',
            s3_interval=args.s3_interval
        ) if args.save_logs else None,
        loggers.VideoLogger(
            env,
            f'{PATH}/experiments/{args.save_folder}/videos/'
        ) if args.save_video else None,
        loggers.HeatMapLogger(
            env,
            save_folder=f'{PATH}/experiments/{args.save_folder}/heatmaps/',
            interval=args.heatmap_interval
        ) if args.save_heatmap else None,
    ]
    callbacks_to_run = [c for c in callbacks_to_run if c]
    callback_list = multigrid.core.CallbackList(callbacks_to_run)
    callback_list.on_train_begin()
    torch.autograd.set_detect_anomaly(True)
    multiagent_run = core.MultiAgentRun(
        env=env,
        models=models,
        interaction_handler=interaction_handler,
        callbacks=callback_list,
        trainers=rl_trainer,
        warm_start=args.warm_start,
        initial_steps=resume_data.current_steps,
        total_steps=args.total_steps,
        initial_episodes=resume_data.current_episodes,
        total_episodes=args.total_episodes
    )
    multiagent_run.run()
    print('-'*20)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser = arguments.add_common_arguments(parser)
    parser = arguments.add_training_arguments(parser)
    parser = arguments.add_model_arguments(parser)
    parser = arguments.add_observation_arguments(parser)
    parser = arguments.add_snake_env_arguments(parser)
    parser = arguments.add_laser_tag_env_arguments(parser)
    parser = arguments.add_render_arguments(parser)
    parser = arguments.add_output_arguments(parser)
    args = parser.parse_args()

    # Write args file
    argsfile = f'{PATH}/experiments/{args.save_folder}/args.txt'
    os.makedirs(os.path.split(argsfile)[0], exist_ok=True)
    if not os.path.exists(argsfile):
        with open(f'{PATH}/experiments/{args.save_folder}/args.txt', 'w') as f:
            print('python ' + ' '.join(sys.argv[1:]), file=f)

    if args.dtype == 'float':
        args.dtype = torch.float
    elif args.dtype == 'half':
        args.dtype = torch.half
    else:
        raise RuntimeError

    if args.repeat is not None and args.n_repeats is not None:
        raise ValueError('Can\'t specify both the number of repeats and a particular repeat number')
    elif args.repeat is not None and args.n_repeats is None:
        repeats = [args.repeat, ]
    elif args.repeat is None and args.n_repeats is not None:
        repeats = list(range(args.n_repeats))
    else:
        repeats = [0, ]

    n_gpus = torch.cuda.device_count()

    # Choose devices
    if len(repeats) == 1:
        devices = [args.device, ]
    else:
        # Assuming multi GPU training here
        devices = [f'cuda:{i}' for i in list(range(n_gpus)) * math.ceil(len(repeats) / n_gpus)][:len(repeats)]

    pool = multiprocessing.Pool(args.n_processes)
    worker_args = [(r, d, a) for r, d, a in zip(repeats, devices, [args, ]*args.n_repeats)]
    print([(r, d) for r, d, _ in worker_args])
    try:
        pool.starmap(worker, worker_args)
    except KeyboardInterrupt:
        raise Exception
    finally:
        pool.close()
