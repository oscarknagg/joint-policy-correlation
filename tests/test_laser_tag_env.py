import unittest
import pytest
import torch
from time import sleep
import matplotlib.pyplot as plt
from torch.distributions import Categorical

from multigrid.envs import LaserTag
from multigrid.envs.laser_tag import maps
from multigrid.envs.laser_tag.map_generators import MapFromString, MapPool
from multigrid import observations
from multigrid.agents import GRUAgent
from tests._laser_trajectories import expected_laser_trajectories_0_2, expected_laser_trajectories_1_3
from config import DEFAULT_DEVICE, PATH


render_envs = False
size = 9
render_sleep = 0.4
# render_sleep = 1
torch.random.manual_seed(3)


def get_test_env(num_envs=2):
    # Same as maps.maps.small2 map from the Deepmind paper
    env = LaserTag(num_envs, 2, height=size, width=size, map_generator=MapFromString(maps.small2, DEFAULT_DEVICE),
                   manual_setup=True, colour_mode='fixed', strict=True)
    for i in range(num_envs):
        env.agents[2*i, :, 1, 1] = 1
        env.agents[2*i + 1, :, 7, 7] = 1
        env.orientations = torch.tensor([0, 2]*num_envs, dtype=torch.long, device=env.device, requires_grad=False)

    env.pathing[:, :, 3, 3] = 1
    env.pathing[:, :, 3, 5] = 1
    env.pathing[:, :, 4, 2] = 1
    env.pathing[:, :, 4, 3] = 1
    env.pathing[:, :, 4, 5] = 1
    env.pathing[:, :, 4, 6] = 1
    env.pathing[:, :, 5, 3] = 1
    env.pathing[:, :, 5, 5] = 1
    env.pathing[:, :, :1, :] = 1
    env.pathing[:, :, -1:, :] = 1
    env.pathing[:, :, :, :1] = 1
    env.pathing[:, :, :, -1:] = 1

    env.respawns = torch.zeros((num_envs, 1, size, size), dtype=torch.uint8, device=DEFAULT_DEVICE, requires_grad=False)
    env.respawns[:, :, 1, 1] = 1
    env.respawns[:, :, 1, 7] = 1
    env.respawns[:, :, 7, 1] = 1
    env.respawns[:, :, 7, 7] = 1

    return env


def render(env):
    if render_envs:
        env.render()
        sleep(render_sleep)


def _test_action_sequence(test_fixture, env, all_actions, expected_orientations=None, expected_x=None, expected_y=None,
                          expected_hp=None, expected_reward=None, expected_done=None):
    render(env)

    for i in range(all_actions['agent_0'].shape[0]):
        actions = {
            agent: agent_actions[i] for agent, agent_actions in all_actions.items()
        }

        obs, rewards, dones, info = env.step(actions)
        render(env)
        env.check_consistency()

        if expected_x is not None:
            test_fixture.assertTrue(torch.equal(env.x.cpu(), expected_x[i]))
        if expected_y is not None:
            test_fixture.assertTrue(torch.equal(env.y.cpu(), expected_y[i]))
        if expected_orientations is not None:
            test_fixture.assertTrue(torch.equal(env.orientations.cpu(), expected_orientations[i]))
        if expected_hp is not None:
            test_fixture.assertTrue(torch.equal(env.hp.cpu(), expected_hp[i]))
        if expected_reward is not None:
            test_fixture.assertTrue(torch.equal(env.rewards.cpu(), expected_reward[i]))
        if expected_done is not None:
            test_fixture.assertTrue(torch.equal(env.dones.cpu(), expected_done[i]))

        env.reset()


class TestLaserTag(unittest.TestCase):
    def test_random_actions(self):
        """Tests a very large number of random actions and checks for environment consistency
        instead of any particular expected trajectory."""
        torch.random.manual_seed(0)
        num_envs = 128
        num_steps = 512
        num_agents = 2
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=9,
            behind=2,
            side=4,
            padding_value=127
        )
        env = LaserTag(num_envs=num_envs, num_agents=2, height=9, width=9, verbose=False,
                       map_generator=MapFromString(maps.small2, DEFAULT_DEVICE), observation_fn=obs_fn,
                       render_args={'num_rows': 3, 'num_cols': 3, 'size': 256},
                       device=DEFAULT_DEVICE, strict=True)
        all_actions = {
            f'agent_{i}': torch.randint(env.num_actions, size=(num_steps, num_envs)).long().to(DEFAULT_DEVICE) for i in
            range(num_agents)
        }

        _test_action_sequence(self, env, all_actions)

    @pytest.mark.skip()
    def test_agent_actions(self):
        """Tests a very large number of actions using pretrained agents and checks for environment consistency. This
        complements the random actions test as it tests trajectories that are more likely to occur when training
        agents."""
        pretrained_agents = [
            f'{PATH}/models/jpc-0a/env=laser__n_envs=128__n_agents=2__n_species=2__agent=gru__r=2__lr=0.001__gamma=0.99__update_steps=5__entropy=0.05__laser_tag_map=small2__species=1.pt',
            # f'{PATH}/tests/data/pretrained_gru_agent_0.pt',
            f'{PATH}/models/jpc-0a/env=laser__n_envs=128__n_agents=2__n_species=2__agent=gru__r=2__lr=0.001__gamma=0.99__update_steps=5__entropy=0.05__laser_tag_map=small2__species=1.pt',
            # f'{PATH}/tests/data/pretrained_gru_agent_1.pt',
        ]
        num_envs = 32
        num_steps = 128
        num_agents = 2
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=11,
            behind=2,
            side=6,
            padding_value=127
        )
        env = LaserTag(num_envs=num_envs, num_agents=2, height=9, width=9, verbose=False,
                       map_generator=MapFromString(maps.small2, DEFAULT_DEVICE), observation_fn=obs_fn,
                       render_args={'num_rows': 3, 'num_cols': 3, 'size': 256},
                       device=DEFAULT_DEVICE, strict=True, colour_mode='fixed')

        # Get models
        models = []
        for i in range(num_agents):
            m = GRUAgent(
                num_actions=env.num_actions, num_initial_convs=2, in_channels=3, conv_channels=32,
                num_residual_convs=2, num_feedforward=1, feedforward_dim=64).to(device=DEFAULT_DEVICE, dtype=torch.float)
            m.load_state_dict(torch.load(pretrained_agents[i]))
            models.append(m)

        with torch.no_grad():
            # Setup env
            hidden_states = {f'agent_{i}': torch.zeros((num_envs, 64), device=DEFAULT_DEVICE) for i in range(num_agents)}
            obs = env.reset()

            for i_step in range(num_steps):
                print('-'*20, i_step, '-'*20)
                actions = {}
                for i, (agent, obs) in enumerate(obs.items()):
                    model = models[i]
                    probs_, value_, hidden_states[agent] = model(obs, hidden_states.get(agent))
                    action_distribution = Categorical(probs_)
                    actions[agent] = action_distribution.sample().clone().long()

                obs, reward, done, info = env.step(actions)
                # env.render(env=19)
                # sleep(0.3)
                env.reset(done['__all__'], return_observations=False)
                env.check_consistency()

    def test_basic_movement(self):
        """2 agents rotate completely on the spot then move in a circle."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([1, 1, 1, 1, 3, 2, 3, 2, 3, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        expected_orientations = torch.tensor([
            [1, 2, 3, 0, 0, 3, 3, 2, 2, 1, 1],
            [1, 0, 3, 2, 2, 1, 1, 0, 0, 3, 3],
        ]).t()
        expected_x = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
            [7, 7, 7, 7, 6, 6, 6, 6, 7, 7, 7],
        ]).t()
        expected_y = torch.tensor([
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1],
            [7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 7],
        ]).t()

        _test_action_sequence(self, env, all_actions, expected_orientations, expected_x, expected_y)

    def test_move_backward(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([3, 4]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([3, 4]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [2, 1],
            [6, 7],
        ]).t()
        expected_y = torch.tensor([
            [1, 1],
            [7, 7],
        ]).t()
        expected_orientations = torch.tensor([
            [0, 0],
            [2, 2],
        ]).t()

        _test_action_sequence(self, env, all_actions, expected_orientations, expected_x, expected_y)

    def test_strafe(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([6, 5]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([6, 5]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [1, 1],
            [7, 7],
        ]).t()
        expected_y = torch.tensor([
            [2, 1],
            [6, 7],
        ]).t()
        expected_orientations = torch.tensor([
            [0, 0],
            [2, 2],
        ]).t()

        _test_action_sequence(self, env, all_actions, expected_orientations, expected_x, expected_y)

    def test_wall_pathing(self):
        """Test agents can't walk through walls."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 5, 6, 6, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([4, 3, 0, 3, 2, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_x = torch.tensor([
            [1, 1, 1, 1, 2, 2, 2],
            [7, 6, 6, 5, 5, 5, 5],
        ]).t()
        expected_y = torch.tensor([
            [1, 1, 2, 3, 3, 3, 3],
            [7, 7, 7, 7, 7, 6, 6]
        ]).t()

        _test_action_sequence(self, env, all_actions, None, expected_x, expected_y)

    def test_move_onto_other_agent(self):
        """Test an agent can't move onto another agent."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([3, 3, 3, 3, 3, 3, 3, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        _test_action_sequence(self, env, all_actions)

    def test_move_through_other_agent(self):
        """If both agents are facing directly toward each other then attempt to move forwards they should not move
        anywhere."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([3, 3, 3, 3, 0, 0, 0, 0, 0, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([2, 3, 3, 3, 3, 3, 3, 1, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        _test_action_sequence(self, env, all_actions)

    def test_agents_move_into_same_square(self):
        """Test two agent can't simultaneously choose to move onto the same square."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 3, 3, 3, 3, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([2, 3, 3, 3, 3, 3, 3, 3]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        _test_action_sequence(self, env, all_actions)

    def test_firing_orientations_0_2(self):
        """Tests that laser trajectories are calculated correctly."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([7, 6, 7, 6, 7, 2, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 7, 2, 7, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            # Laser trajectories were verified manually then saved to another file because they are very verbose
            self.assertTrue(torch.equal(env.lasers, expected_laser_trajectories_0_2[i]))
            render(env)

    def test_firing_orientations_1_3(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([9, 7, 5, 7, 5, 7, 5, 7, 5, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 9, 7, 5, 7, 5, 7, 5, 7, 5, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            # Laser trajectories were verified manually then saved to another file because they are very verbose
            self.assertTrue(torch.equal(env.lasers, expected_laser_trajectories_1_3[i]))
            render(env)

    def test_being_hit_orientations_0_2(self):
        """Tests that agent-laser collision is calculated correctly, hp is deducted
        and reward is given."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 7, 7, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 0, 0, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_hp = torch.tensor([
            [2, ] * all_actions['agent_0'].shape[0],
            [2, ] * (all_actions['agent_0'].shape[0] - 2) + [1, 0],
        ]).t()
        expected_reward = torch.tensor([
            [0, ] * (all_actions['agent_0'].shape[0] - 2) + [0, 1],
            [0, ] * (all_actions['agent_0'].shape[0] - 2) + [0, 0],
        ]).t().float()
        expected_done = torch.tensor([
            [0, ] * all_actions['agent_0'].shape[0],
            [0, ] * (all_actions['agent_0'].shape[0] - 1) + [1],
        ]).t().byte()

        _test_action_sequence(self, env, all_actions, expected_hp=expected_hp, expected_reward=expected_reward,
                              expected_done=expected_done)

    def test_being_hit_orientations_1_3(self):
        """Tests that agent-laser collision is calculated correctly, hp is deducted
        and reward is given."""
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([3, 3, 3, 3, 3, 3, 2, 0, 0, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        expected_hp = torch.tensor([
            [2, ] * (all_actions['agent_0'].shape[0] - 3) + [2, 1, 1],
            [2, ] * (all_actions['agent_0'].shape[0] - 3) + [1, 1, 0],
        ]).t()
        expected_reward = torch.tensor([
            [0, ] * (all_actions['agent_0'].shape[0] - 3) + [0, 0, 1],
            [0, ] * (all_actions['agent_0'].shape[0] - 3) + [0, 0, 0],
        ]).t().float()
        expected_done = torch.tensor([
            [0, ] * all_actions['agent_0'].shape[0],
            [0, ] * (all_actions['agent_0'].shape[0] - 1) + [1],
        ]).t().byte()

        _test_action_sequence(self, env, all_actions, expected_hp=expected_hp, expected_reward=expected_reward,
                              expected_done=expected_done)

    def test_cant_shoot_through_agents(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 1, 3, 0, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            if i == 10:
                # This checks that the space behind agent_1 doesn't contain a laser
                self.assertEqual(env.lasers[0, 0, 7, 1].item(), 0)

            render(env)

    def test_death_and_respawn(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)

            env.reset(dones['__all__'])

            render(env)

    def test_random_firing_resolution(self):
        """In a straight fight only one agent should receive reward (at random) and the other should die"""
        torch.random.manual_seed(2)
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 1, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        _test_action_sequence(self, env, all_actions)
        # TODO: Add checks for reward and dones.
        # raise NotImplementedError('Add checks for reward and dones')

    def test_observations(self):
        obs_fn = observations.RenderObservations()
        env = LaserTag(num_envs=1, num_agents=2, height=9, width=9,
                       map_generator=MapFromString(maps.small2, DEFAULT_DEVICE), observation_fn=obs_fn,
                       device=DEFAULT_DEVICE, strict=True)

        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()

    def test_partial_observations(self):
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=17,
            behind=2,
            side=10,
            padding_value=127
        )
        env = get_test_env(num_envs=2)

        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()

        if render_envs:
            env.render()
            sleep(5)

    def test_create_envs(self):
        obs_fn = observations.RenderObservations()
        env = LaserTag(num_envs=16, num_agents=2, height=9, width=9,
                       map_generator=MapFromString(maps.small2, DEFAULT_DEVICE), observation_fn=obs_fn,
                       device=DEFAULT_DEVICE, strict=True)
        env.check_consistency()

    def test_render(self):
        env = get_test_env()

        if render_envs:
            env.render()
            sleep(5)


class TestSmall3(unittest.TestCase):
    def test_firing_orientations_1_3(self):
        env = LaserTag(num_envs=1, num_agents=2, height=9, width=16, map_generator=MapFromString(maps.small3, DEFAULT_DEVICE),
                       device=DEFAULT_DEVICE, colour_mode='fixed', strict=True)

        env.agents = torch.zeros_like(env.agents)
        env.agents[0, 0, 1, 1] = 1
        env.agents[1, 0, -2, -2] = 1
        env.orientations[0] = 3
        env.orientations[1] = 1

        all_actions = {
            'agent_0': torch.tensor([7, 0, 7, 0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3, 3, 6, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 7, 5, 7, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            print('-'*20, i, '-'*20)
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            # Laser trajectories were verified manually then saved to another file because they are very verbose
            # self.assertTrue(torch.equal(env.lasers, expected_laser_trajectories_1_3[i]))

            # print(env.lasers)

            render(env)

    def test_firing_orientations_0_2(self):
        env = LaserTag(num_envs=1, num_agents=2, height=9, width=16, map_generator=MapFromString(maps.small3, DEFAULT_DEVICE),
                       device=DEFAULT_DEVICE, colour_mode='fixed', strict=True)

        env.agents = torch.zeros_like(env.agents)
        env.agents[0, 0, 1, 14] = 1
        env.agents[1, 0, -2, -2] = 1
        env.orientations[0] = 0
        env.orientations[1] = 1

        all_actions = {
            'agent_0': torch.tensor([7, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([3, ]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }

        render(env)

        for i in range(all_actions['agent_0'].shape[0]):
            print('-'*20, i, '-'*20)
            actions = {
                agent: agent_actions[i] for agent, agent_actions in all_actions.items()
            }

            observations, rewards, dones, info = env.step(actions)
            # Laser trajectories were verified manually then saved to another file because they are very verbose
            # self.assertTrue(torch.equal(env.lasers, expected_laser_trajectories_1_3[i]))

            # print(env.lasers)

            render(env)

    def test_partial_observations(self):
        torch.random.manual_seed(0)
        obs_fn = observations.FirstPersonCrop(
            first_person_rotation=True,
            in_front=13,
            behind=2,
            side=6,
            padding_value=127
        )

        # env = LaserTag(num_envs=1, num_agents=2, height=9, width=9, map_generator=MapFromString(maps.small2, DEFAULT_DEVICE),
        #                device=DEFAULT_DEVICE, colour_mode='fixed', strict=True)

        # env = LaserTag(num_envs=1, num_agents=2, height=9, width=16, map_generator=MapFromString(maps.small3, DEFAULT_DEVICE),
        #                device=DEFAULT_DEVICE, colour_mode='fixed', strict=True)

        env = LaserTag(num_envs=1, num_agents=2, height=14, width=22, map_generator=MapFromString(maps.small4, DEFAULT_DEVICE),
                       device=DEFAULT_DEVICE, colour_mode='fixed', strict=True)

        render_envs = True
        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()

        if render_envs:
            env.render()
            sleep(5)


# class TestMaps(unittest.TestCase):
#     def test_map_pool(self):
#         map_pool = [
#             MapFromString(maps.small2, DEFAULT_DEVICE),
#             MapFromString(maps.small2b, DEFAULT_DEVICE),
#         ]
#
#         map_generator = MapPool(map_pool)
#
#         new_maps = map_generator.generate(5)
