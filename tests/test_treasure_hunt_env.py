import unittest
import torch
from time import sleep
import matplotlib.pyplot as plt

from multigrid.envs import TreasureHunt
from multigrid.envs.treasure_hunt import maps
from multigrid.envs.maps import parse_mapstring, FixedMapGenerator
from multigrid import observations
from config import DEFAULT_DEVICE


DEFAULT_DEVICE = 'cpu'
render_envs = True
size = 9
render_sleep = 0.4
# render_sleep = 1
torch.random.manual_seed(3)


def get_test_env(num_envs=2):
    # Same as maps.maps.small2 map from the Deepmind paper
    env = TreasureHunt(num_envs, 2, height=size, width=size,
                       map_generator=FixedMapGenerator(parse_mapstring('treasure_hunt-small2'), DEFAULT_DEVICE),
                       manual_setup=True, colour_mode='fixed', strict=True, device=DEFAULT_DEVICE)

    for i in range(num_envs):
        env.agents[2*i, :, 1, 1] = 1
        env.agents[2*i + 1, :, 1, 7] = 1
        env.orientations = torch.tensor([0, 1]*num_envs, dtype=torch.long, device=env.device, requires_grad=False)

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

    env.treasure = torch.zeros((num_envs, 1, size, size), dtype=torch.uint8, device=DEFAULT_DEVICE, requires_grad=False)
    env.treasure[:, :, 1, 3] = 1
    env.treasure[:, :, 3, 6] = 1
    env.treasure[:, :, 5, 2] = 1
    env.treasure[:, :, 7, 5] = 1

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
    def test_dig(self):
        env = get_test_env(num_envs=1)
        all_actions = {
            'agent_0': torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
            'agent_1': torch.tensor([0, 2, 3, 3, 3, 3, 3, 3, 1, 7, 7, 0]).unsqueeze(1).long().to(DEFAULT_DEVICE),
        }
        _test_action_sequence(self, env, all_actions)

    def test_observations(self):
        obs_fn = observations.RenderObservations()
        env = TreasureHunt(num_envs=1, num_agents=2, height=9, width=9,
                           map_generator=FixedMapGenerator(parse_mapstring('treasure_hunt-small2'), DEFAULT_DEVICE),
                           observation_fn=obs_fn, device=DEFAULT_DEVICE, strict=True)

        agent_obs = obs_fn.observe(env)
        for agent, obs in agent_obs.items():
            if render_envs:
                obs_npy = obs.permute((2, 3, 1, 0))[:, :, :, 0].cpu().numpy()
                plt.imshow(obs_npy)
                plt.show()
