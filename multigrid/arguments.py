import argparse
from torch import nn, optim
from typing import List, Optional, Dict
import os
import torch

from multigrid.envs import LaserTag, Slither, TreasureHunt
from multigrid.envs.maps import parse_mapstring, MapPool, Random, maps_from_file, FixedMapGenerator
from multigrid.observations import ObservationFunction
from multigrid import rl
from multigrid import utils
from multigrid import agents
from config import PATH, INPUT_CHANNELS


def get_bool(input_string: str) -> bool:
    return input_string.lower()[0] == 't'


def get_dtype(dtype) -> torch.dtype:
    if dtype == 'float':
        dtype = torch.float
    elif dtype == 'half':
        dtype = torch.half
    else:
        raise ValueError('Unrecognised data type.')

    return dtype


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--env', type=str)
    parser.add_argument('--env-map', nargs='+', type=str, default='random')
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
    parser.add_argument('--strict', default=False, type=get_bool,
                        help='Whether to raise an exception if an env experiences an inconsistency.')

    return parser


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--train', default=True, type=get_bool)
    parser.add_argument('--batch-norm', default=False, type=get_bool)
    parser.add_argument('--n-pool', default=1, type=int)
    parser.add_argument('--pool-steps', default=1000, type=int)
    parser.add_argument('--value-loss-coeff', default=1.0, type=float)
    parser.add_argument('--entropy', default=0.01, type=float)
    parser.add_argument('--diversity', default=0.00, type=float)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--mask-dones', default=True, type=get_bool, help='Removes deaths from training trajectories.')
    parser.add_argument('--train-algo', default='a2c', type=str)
    parser.add_argument('--ppo-eta-clip', default=0.1, type=float)
    parser.add_argument('--ppo-epochs', default=3, type=int)
    parser.add_argument('--ppo-batch', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight-decay', default=0, type=float)
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
    parser.add_argument('--colour-mode', type=str, default='fixed')
    return parser


def add_laser_tag_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--pathing-file', type=str)
    parser.add_argument('--respawn-file', type=str)
    # Random map generation arguments
    parser.add_argument('--maze-complexity', type=float)
    parser.add_argument('--maze-density', type=float)
    parser.add_argument('--n-respawns', type=int)
    parser.add_argument('--n-maps', type=int, default=None)
    return parser


def add_treasure_hunt_env_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--treasure-file', type=str)
    parser.add_argument('--treasure-refresh', type=int, default=20)
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
    parser.add_argument('--model-interval', default=500000, type=int)
    parser.add_argument('--s3-interval', default=500000, type=int)
    parser.add_argument('--s3-bucket', default=None, type=str)
    parser.add_argument('--heatmap-interval', default=None, type=int)
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
        if len(args.env_map) == 1:
            if args.env_map[0] == 'random':
                # Generate n_maps random mazes to select from at random during each map reset
                map_generator = Random(args.n_respawns, args.height, args.width, args.maze_complexity,
                                       args.maze_density, args.device)
            elif args.env_map[0] == 'from_file':
                maps = maps_from_file(args.pathing_file, args.respawn_file, args.device, args.n_maps)
                map_generator = MapPool(maps)
            else:
                # Single fixed map
                map_generator = FixedMapGenerator(parse_mapstring(args.env_map[0]), device)
        else:
            fixed_maps = [parse_mapstring(m) for m in args.env_map]
            map_generator = MapPool(fixed_maps)

        env = LaserTag(num_envs=args.n_envs, num_agents=args.n_agents, height=args.height, width=args.width,
                       observation_fn=observation_fn, colour_mode=args.colour_mode,
                       map_generator=map_generator, device=device, render_args=render_args, strict=args.strict)
    elif args.env == 'treasure':
        if len(args.env_map) == 1:
            if args.env_map[0] == 'random':
                # Generate n_maps random mazes to select from at random during each map reset
                map_generator = Random(args.n_respawns, args.height, args.width, args.maze_complexity,
                                       args.maze_density, args.device)
            elif args.env_map[0] == 'from_file':
                maps = maps_from_file(args.pathing_file, args.respawn_file, args.device, args.n_maps,
                                      other_tensors={'treasure': args.treasure_file})
                map_generator = MapPool(maps)
            else:
                # Single fixed map
                map_generator = FixedMapGenerator(parse_mapstring(args.env_map[0]), device)
        else:
            fixed_maps = [parse_mapstring(m) for m in args.env_map]
            map_generator = MapPool(fixed_maps)

        env = TreasureHunt(num_envs=args.n_envs, num_agents=args.n_agents, height=args.height, width=args.width,
                           observation_fn=observation_fn, colour_mode=args.colour_mode, treasure_refresh_rate=args.treasure_refresh,
                           map_generator=map_generator, device=device, render_args=render_args, strict=args.strict)

    elif args.env == 'asymmetric':
        raise NotImplementedError
    else:
        raise ValueError('Unrecognised environment')

    return env


def get_models(args: argparse.Namespace, num_agent_types: int, num_actions: int, device: str) -> Dict[str, List[nn.Module]]:
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

    # if args.agent_location is not None:
    models: Dict[str, List[nn.Module]] = {f'agent_{i}': [] for i in range(num_agent_types)}
    for i_agent_type in range(num_agent_types):
        for i_model in range(args.n_pool):
            # Check for existence of model file
            if args.agent_location is None or args.agent_location == []:
                specified_model_file = False
            else:
                model_path = args.agent_location[i_agent_type][i_model]
                model_relative_path = os.path.join(PATH, 'models', args.agent_location[i_agent_type][i_model])
                if os.path.exists(model_path) or os.path.exists(model_relative_path):
                    specified_model_file = True
                else:
                    specified_model_file = False

            # Create model class
            if args.agent_type in ('lstm', 'gru'):
                models[f'agent_{i_agent_type}'].append(
                    agents.RecurrentAgent(recurrent_module=args.agent_type, num_actions=num_actions,
                                          in_channels=INPUT_CHANNELS, channels=16, fc_size=32, hidden_size=32,
                                          batch_norm=args.batch_norm).to(device=device, dtype=args.dtype)
                )
            elif args.agent_type == 'random':
                models[f'agent_{i_agent_type}'].append(agents.RandomAgent(num_actions=num_actions, device=device))
            else:
                raise ValueError('Unrecognised agent type.')

            # Load state dict if the model file(s) have been specified
            if specified_model_file:
                print('Reloading agent={} model={} from location {}'.format(
                    i_agent_type,
                    i_model,
                    args.agent_location[i_agent_type][i_model]
                ))
                models[f'agent_{i_agent_type}'][i_model].load_state_dict(
                    utils.load_state_dict(args.agent_location[i_agent_type][i_model])
                )

            # Add agent attributes
            models[f'agent_{i_agent_type}'][-1].agent_id = i_agent_type
            models[f'agent_{i_agent_type}'][-1].pool_id = i_model
            models[f'agent_{i_agent_type}'][-1].train_steps = 0
            models[f'agent_{i_agent_type}'][-1].train_episodes = 0
            models[f'agent_{i_agent_type}'][-1].last_saved_steps = 0
            models[f'agent_{i_agent_type}'][-1].num_checkpoints = 0

    if args.train:
        for agent, _models in models.items():
            for m in _models:
                m.train()
    else:
        torch.no_grad()
        for agent, _models in models.items():
            for m in _models:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    return models


def get_trainers(args: argparse.Namespace,
                 num_agent_types: int,
                 models: Dict[str, List[nn.Module]]) -> Optional[Dict[str, List[rl.core.SingleAgentTrainer]]]:
    if args.train:
        trainers = {f'agent_{i}': [] for i in range(num_agent_types)}
        for i_agent_type, (agent_type, _models) in enumerate(models.items()):
            for i, m in enumerate(_models):
                if args.train_algo == 'a2c':
                    trainers[agent_type].append(
                        rl.A2CTrainer(
                            agent_id=f'agent_{i_agent_type}',
                            model=m,
                            update_steps=args.update_steps,
                            optimizer=optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                            a2c=rl.ActorCritic(gamma=args.gamma, normalise_returns=args.norm_returns, dtype=args.dtype,
                                               use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda),
                            max_grad_norm=args.max_grad_norm,
                            value_loss_coeff=args.value_loss_coeff,
                            entropy_loss_coeff=args.entropy,
                            mask_dones=args.mask_dones
                        )
                    )
                elif args.train_algo == 'ppo':
                    trainers[agent_type].append(
                        rl.PPOTrainer(
                            agent_id=f'agent_{i_agent_type}',
                            model=m,
                            update_steps=args.update_steps,
                            optimizer=optim.Adam(m.parameters(), lr=args.lr, weight_decay=args.weight_decay),
                            a2c=rl.ActorCritic(gamma=args.gamma, normalise_returns=args.norm_returns,
                                               dtype=args.dtype,
                                               use_gae=args.gae_lambda is not None, gae_lambda=args.gae_lambda),
                            max_grad_norm=args.max_grad_norm,
                            value_loss_coeff=args.value_loss_coeff,
                            entropy_loss_coeff=args.entropy,
                            mask_dones=args.mask_dones,
                            diversity_loss_coeff=args.diversity,
                            # PPO specific arguments
                            batch_size=args.ppo_batch,
                            epochs=args.ppo_epochs,
                            eta_clip=args.ppo_eta_clip,
                        )
                    )
                else:
                    raise ValueError('Unrecognised algorithm.')
    else:
        trainers = None

    return trainers
