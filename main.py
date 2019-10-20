"""Entry point for training, transferring and visualising agents."""
import argparse
import math
import sys
import os

import torch
from torch import multiprocessing

from multigrid import arguments
from multigrid import resume
from multigrid import observations
from multigrid.pool import MultiAgentPoolRun
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
    # Do a warm-start if we're resuming and there is no warm start specified
    args.warm_start = 500 if (resume_data.resume and args.warm_start == 0) else args.warm_start

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
    models = arguments.get_models(args, env.num_agents, env.num_actions, device)
    trainers = arguments.get_trainers(args, env.num_agents, models)

    torch.autograd.set_detect_anomaly(True)
    multiagent_run = MultiAgentPoolRun(
        env=env,
        n_pool=args.n_pool,
        models=models,
        trainers=trainers,
        mask_dones=args.mask_dones,
        warm_start=args.warm_start,
        repeat_number=repeat,
        schedule_steps=args.pool_steps,
        initial_steps=resume_data.current_steps,
        total_steps=args.total_steps,
        initial_episodes=resume_data.current_episodes,
        total_episodes=args.total_episodes,
        args=args
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
    parser = arguments.add_harvest_env_arguments(parser)
    parser = arguments.add_render_arguments(parser)
    parser = arguments.add_output_arguments(parser)
    args = parser.parse_args()

    # Write args file
    if args.save_model or args.save_logs or args.save_video or args.save_heatmap:
        argsfile = f'{PATH}/experiments/{args.save_folder}/args.txt'
        os.makedirs(os.path.split(argsfile)[0], exist_ok=True)
        if not os.path.exists(argsfile):
            with open(f'{PATH}/experiments/{args.save_folder}/args.txt', 'w') as f:
                print('python ' + ' '.join(sys.argv[1:]), file=f)

    args.dtype = arguments.get_dtype(args.dtype)

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

    # Only use multiprocessing if specified as we get better error traces
    # without multiprocessing.
    worker_args = [(r, d, a) for r, d, a in zip(repeats, devices, [args, ]*args.n_repeats)]
    if args.n_processes > 1:
        pool = multiprocessing.Pool(args.n_processes, maxtasksperchild=1)
        print([(r, d) for r, d, _ in worker_args])
        try:
            pool.starmap(worker, worker_args)
        except KeyboardInterrupt:
            raise Exception
        finally:
            pool.close()
    else:
        for run_args in worker_args:
            worker(*run_args)

