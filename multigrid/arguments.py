import argparse
from torch import nn, optim
from typing import List, Optional
import os
import torch

from multigrid.envs import LaserTag, Slither
from multigrid.envs.laser_tag.map_generators import MapFromString, MapPool
from multigrid.observations import ObservationFunction
from multigrid import rl
from multigrid import utils
from multigrid import agents
from config import PATH, INPUT_CHANNELS


def get_bool(input_string: str) -> bool:
    return input_string.lower()[0] == 't'


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--env', type=str)
    parser.add_argument('--n-envs', type=int)
    parser.add_argument('--n-agents', type=int)
    parser.add_argument('--n-species', type=int, default=1)
    parser.add_argument('--height', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--warm-start', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--dtype', type=str, default='float')
    parser.add_argument('--repeat', default=None, type=int, help='Repeat number if running a single repeat.')
    parser.add_argument('--n-repeats', default=1, type=int, help='Number of repeats to run')
    parser.add_argument('--n-processes', default=1, type=int)
    parser.add_argument('--resume-mode', default='local', type=str)
    return parser


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--diayn', default=0, type=float)
    parser.add_argument('--value-loss-coeff', default=0.5, type=float)
    parser.add_argument('--entropy', default=0.01, type=float)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--coord-conv', default=True, type=get_bool)
    parser.add_argument('--mask-dones', default=True, type=get_bool, help='Removes deaths from training trajectories.')
    parser.add_argument('--train', default=True, type=get_bool)
    parser.add_argument('--train-algo', default='a2c', type=str)
    parser.add_argument('--ppo-eta-clip', default=0.1, type=float)
    parser.add_argument('--ppo-epochs', default=3, type=int)
    parser.add_argument('--ppo-batch', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--gae-lambda', default=None, type=float)
    parser.add_argument('--update-steps', default=5, type=int)
    parser.add_argument('--total-steps', default=float('inf'), type=float)
    parser.add_argument('--total-episodes', default=float('inf'), type=float)
    parser.add_argument('--norm-advantages', default=True, type=get_bool)
    parser.add_argument('--norm-returns', default=False, type=get_bool)
    parser.add_argument('--share-backbone', default=False, type=get_bool)
    return parser


def add_model_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--agent-type', type=str, default='gru')
    parser.add_argument('--agent-location', type=str, nargs='+')
    return parser


def add_observation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--obs-h', type=int)
    parser.add_argument('--obs-w', type=int)
    parser.add_argument('--obs-rotate', type=get_bool)
    parser.add_argument('--obs-in-front', type=int)
    parser.add_argument('--obs-behind', type=int)
    parser.add_argument('--obs-side', type=int)
    return parser


def add_snake_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--boost', default=True, type=get_bool)
    parser.add_argument('--boost-cost', type=float, default=0.25)
    parser.add_argument('--food-on-death', type=float, default=0.33)
    parser.add_argument('--reward-on-death', type=float, default=-1)
    parser.add_argument('--food-mode', type=str, default='random_rate')
    parser.add_argument('--food-rate', type=float, default=3e-4)
    parser.add_argument('--respawn-mode', type=str, default='any')
    parser.add_argument('--colour-mode', type=str, default='random')
    return parser


def add_laser_tag_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--laser-tag-map', nargs='+', type=str, default='random')
    return parser


def add_render_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--render', default=False, type=lambda x: x.lower()[0] == 't')
    parser.add_argument('--render-window-size', default=256, type=int)
    parser.add_argument('--render-cols', default=1, type=int)
    parser.add_argument('--render-rows', default=1, type=int)
    parser.add_argument('--fps', default=12, type=int)
    return parser


def add_output_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--print-interval', default=1000, type=int)
    parser.add_argument('--log-interval', default=1, type=int)
    parser.add_argument('--model-interval', default=1000, type=int)
    parser.add_argument('--s3-interval', default=None, type=int)
    parser.add_argument('--s3-bucket', default=None, type=str)
    parser.add_argument('--heatmap-interval', default=1, type=int)
    parser.add_argument('--save-folder', type=str, default=None)
    parser.add_argument('--save-location', type=str, default=None)
    parser.add_argument('--save-model', default=True, type=get_bool)
    parser.add_argument('--save-logs', default=True, type=get_bool)
    parser.add_argument('--save-video', default=False, type=get_bool)
    parser.add_argument('--save-heatmap', default=False, type=get_bool)
    return parser


def get_env(args: argparse.Namespace, observation_fn: ObservationFunction, device: str):
    if args.env is None:
        raise ValueError('args.env is None.')
    render_args = {
        'size': args.render_window_size,
        'num_rows': args.render_rows,
        'num_cols': args.render_cols,
    }
    if args.env == 'snake':
        env = Slither(num_envs=args.n_envs, num_agents=args.n_agents, food_on_death_prob=args.food_on_death,
                      height=args.height, width=args.width, device=device, render_args=render_args,
                      boost=args.boost,
                      boost_cost_prob=args.boost_cost, food_rate=args.food_rate,
                      respawn_mode=args.respawn_mode, food_mode=args.food_mode, observation_fn=observation_fn,
                      reward_on_death=args.reward_on_death, agent_colours=args.colour_mode)
    elif args.env == 'laser':
        if len(args.laser_tag_map) == 1:
            map_generator = MapFromString(args.laser_tag_map[0], device)
        else:
            fixed_maps = [MapFromString(m, device) for m in args.laser_tag_map]
            map_generator = MapPool(fixed_maps)

        env = LaserTag(num_envs=args.n_envs, num_agents=args.n_agents, height=args.height, width=args.width,
                       observation_fn=observation_fn, colour_mode=args.colour_mode,
                       map_generator=map_generator, device=device, render_args=render_args)
    elif args.env == 'cooperative':
        raise NotImplementedError
    elif args.env == 'asymmetric':
        raise NotImplementedError
    else:
        raise ValueError('Unrecognised environment')

    return env


def get_models(args: argparse.Namespace, num_actions: int, device: str) -> List[nn.Module]:
    num_models = args.n_species

    # Quick hack to make it easier to input all of the species trained in one particular experiment
    if args.agent_location is not None:
        if len(args.agent_location) == 1:
            species_0_path = args.agent_location[0] + '__species=0.pt'
            species_0_relative_path = os.path.join(PATH, args.agent_location[0]) + '__species=0.pt'
            if os.path.exists(species_0_path) or os.path.exists(species_0_relative_path):
                agent_path = args.agent_location[0] if species_0_path else os.path.join(PATH, 'models',
                                                                                        args.agent_location)
                args.agent_location = [agent_path + f'__species={i}.pt' for i in range(args.n_species)]
            else:
                args.agent_location = [args.agent_location[0], ] * args.n_agents

    models: List[nn.Module] = []
    for i in range(num_models):
        # Check for existence of model file
        if args.agent_location is None or args.agent_location == []:
            specified_model_file = False
        else:
            model_path = args.agent_location[i]
            model_relative_path = os.path.join(PATH, 'models', args.agent_location[i])
            if os.path.exists(model_path) or os.path.exists(model_relative_path):
                specified_model_file = True
            else:
                specified_model_file = False

        # Create model class
        if args.agent_type == 'conv':
            models.append(
                agents.ConvAgent(
                    num_actions=num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                    num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=device,
                                                                                    dtype=args.dtype)
            )
        elif args.agent_type == 'gru':
            models.append(
                agents.GRUAgent(
                    num_actions=num_actions, num_initial_convs=2, in_channels=INPUT_CHANNELS, conv_channels=32,
                    num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=device,
                                                                                    dtype=args.dtype)
            )
        elif args.agent_type == 'random':
            models.append(agents.RandomAgent(num_actions=num_actions, device=device))
        else:
            raise ValueError('Unrecognised agent type.')

        # Load state dict if the model file(s) have been specified
        if specified_model_file:
            print('Reloading agent {} from location {}'.format(i, args.agent_location[i]))
            models[i].load_state_dict(
                utils.load_state_dict(args.agent_location[i])
            )

    if args.train:
        for m in models:
            m.train()
    else:
        torch.no_grad()
        for m in models:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return models


def get_trainers(args: argparse.Namespace, models: List[nn.Module]) -> Optional[rl.core.MultiAgentTrainer]:
    if args.train:
        if args.train_algo == 'a2c':
            a2c_trainers = []
            for i, m in enumerate(models):
                a2c_trainers.append(
                    rl.A2CTrainer(
                        agent_id=f'agent_{i}',
                        model=m,
                        update_steps=args.update_steps,
                        optimizer=optim.Adam(m.parameters(), lr=args.lr, weight_decay=0),
                        a2c=rl.ActorCritic(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=args.dtype,
                                           use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda),
                        max_grad_norm=args.max_grad_norm,
                        value_loss_coeff=args.value_loss_coeff,
                        entropy_loss_coeff=args.entropy,
                        mask_dones=args.mask_dones
                    )
                )
            rl_trainer = rl.core.IndependentTrainer(a2c_trainers)
        elif args.train_algo == 'ppo':
            ppo_trainers = []
            for i, m in enumerate(models):
                ppo_trainers.append(
                    rl.PPOTrainer(
                        agent_id=f'agent_{i}',
                        model=m,
                        update_steps=args.update_steps,
                        optimizer=optim.Adam(m.parameters(), lr=args.lr, weight_decay=0),
                        a2c=rl.ActorCritic(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=args.dtype,
                                           use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda),
                        max_grad_norm=args.max_grad_norm,
                        value_loss_coeff=args.value_loss_coeff,
                        entropy_loss_coeff=args.entropy,
                        mask_dones=args.mask_dones,
                        # PPO specific arguments
                        batch_size=args.ppo_batch,
                        epochs=args.ppo_epochs,
                        eta_clip=args.ppo_eta_clip,
                    )
                )
            rl_trainer = rl.core.IndependentTrainer(ppo_trainers)
        else:
            raise ValueError('Unrecognised algorithm.')
    else:
        rl_trainer = None

    return rl_trainer
