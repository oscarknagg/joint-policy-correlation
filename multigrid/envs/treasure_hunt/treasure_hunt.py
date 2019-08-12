import torch
from typing import Dict, Optional, Tuple
import torch.nn.functional as F
from time import time
import warnings

from multigrid._filters import ORIENTATION_FILTERS
from multigrid.utils import drop_duplicates, get_coords
from multigrid.core import MultiagentVecEnv, check_multi_vec_env_actions
from multigrid.dynamics import move_pixels
from multigrid.envs.maps import MapGenerator
from multigrid.observations import ObservationFunction, RenderObservations
from config import DEFAULT_DEVICE, EPS


class TreasureHunt(MultiagentVecEnv):
    """Co-operative environment."""
    no_op = 0
    rotate_right = 1
    rotate_left = 2
    move_forward = 3
    move_back = 4
    move_right = 5
    move_left = 6
    dig = 7
    move_forward_and_turn_right = 8
    move_forward_and_turn_left = 9

    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 12
    }

    def __init__(self,
                 num_envs: int,
                 num_agents: int,
                 height: int,
                 width: int,
                 map_generator: MapGenerator,
                 strict: bool = False,
                 verbose: int = 0,
                 colour_mode: str = 'fixed',
                 observation_fn: ObservationFunction = RenderObservations(),
                 manual_setup: bool = False,
                 initial_hp: int = 2,
                 render_args: dict = None,
                 env_lifetime: int = 1000,
                 dtype: torch.dtype = torch.float,
                 device: str = DEFAULT_DEVICE):
        super(TreasureHunt, self).__init__(num_envs, num_agents, height, width, dtype, device)
        self.strict = strict
        self.verbose = verbose
        self.map_generator = map_generator
        self.colour_mode = colour_mode
        self.observation_fn = observation_fn
        self.initial_hp = initial_hp
        self.max_env_lifetime = env_lifetime
        self.num_actions = 10

        if colour_mode == 'random':
            self.agent_colours = self._get_n_colours(num_envs*num_agents)
        elif colour_mode == 'fixed':
            fixed_colours = torch.tensor([
                [192, 0, 0],  # Red
                [0, 0, 192],  # Blue
                [0, 192, 0],  # Green
            ], device=device, dtype=torch.short)
            self.agent_colours = fixed_colours[:self.num_agents].repeat((self.num_envs, 1))
        else:
            raise ValueError('colour_mode not recognised.')

        if render_args is None:
            self.render_args = {'num_rows': 1, 'num_cols': 1, 'size': 256}
        else:
            self.render_args = render_args

        # Environment tensors
        self.env_lifetimes = torch.zeros((num_envs), dtype=torch.long, device=self.device, requires_grad=False)

        self.has_dug = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)
        if not manual_setup:
            self.agents, self.orientations, self.dones, self.hp, self.pathing, self.respawns, \
                self.treasure = self._create_envs(num_envs)
        else:
            self.agents = torch.zeros((num_envs*num_agents, 1, height, width), device=device, requires_grad=False)
            self.orientations = torch.zeros((num_envs * num_agents), dtype=torch.long, device=self.device, requires_grad=False)
            self.dones = torch.zeros((num_envs*num_agents), dtype=torch.uint8, device=device, requires_grad=False)
            self.hp = torch.ones((num_envs * num_agents), dtype=torch.long, device=device, requires_grad=False) * self.initial_hp
            self.pathing = torch.zeros((num_envs, 1, height, width), device=device, requires_grad=False, dtype=torch.uint8)
            self.respawns = torch.zeros((num_envs, 1, height, width), device=device, requires_grad=False, dtype=torch.uint8)
            self.treasure = torch.zeros((num_envs, 1, height, width), device=device, requires_grad=False, dtype=torch.uint8)

        # Environment outputs
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

    def _log(self, msg: str):
        if self.verbose > 0:
            print(msg)

    def _other_agents(self) -> torch.Tensor:
        """Calculate positions of other agents with respect to each agent.

        Returns:
            other_agents: A boolean Tensor of shape (num_envs*num_agents, 1, height, width) where each sub-Tensor
                along index 0 is a map of other agents in an environment from the point of view of that particular
                agent.
        """
        t0 = time()
        other_agents = self.agents.view(self.num_envs, self.num_agents, self.height, self.width) \
            .sum(dim=1, keepdim=True).repeat_interleave(self.num_agents, 0) \
            .sub(self.agents) \
            .gt(0.5)
        self._log(f'Other agents: {1000 * (time() - t0)}ms')
        return other_agents

    def step(self, actions: Dict[str, torch.Tensor]) -> (Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict):
        check_multi_vec_env_actions(actions, self.num_envs, self.num_agents)
        info = {}
        actions = torch.stack([v for k, v in actions.items()]).t().flatten()

        # Reset stuff
        self.rewards = torch.zeros(self.num_envs * self.num_agents, dtype=torch.float, device=self.device)
        self.dones = torch.zeros(self.num_envs * self.num_agents, dtype=torch.uint8, device=self.device)

        # Movement
        has_moved = actions == self.move_forward
        has_moved |= actions == self.move_back
        has_moved |= actions == self.move_right
        has_moved |= actions == self.move_left
        has_moved |= actions == self.move_forward_and_turn_right
        has_moved |= actions == self.move_forward_and_turn_left
        if torch.any(has_moved):
            # Calculate movement directions
            # Default movement is in the orientation direction
            directions = self.orientations.clone()
            # Backwards is reverse of the orientation direction
            directions[actions == self.move_back] = directions[actions == self.move_back] + 2
            # Right is orientation +1
            directions[actions == self.move_right] = directions[actions == self.move_right] + 1
            # Left is orientation -1
            directions[actions == self.move_left] = directions[actions == self.move_left] + 3
            # Keep in range(4)
            directions.fmod_(4)

            resolution_order = torch.randperm(self.num_agents, device=self.device).repeat(self.num_envs)
            for i_resolve in range(self.num_agents):
                resolve_these = resolution_order == i_resolve
                currently_moving = has_moved & resolve_these

                if not torch.any(currently_moving):
                    continue

                # Keep the original positions so we can reset if an agent does a move
                # that's disallowed by pathing.
                original_agents = self.agents.clone()

                t0 = time()
                self.agents[currently_moving] = move_pixels(self.agents[currently_moving], directions[currently_moving])
                self.agents.round_()  # Stops numerical errors accumulating
                self._log(f'Movement: {1000 * (time() - t0)}ms')

                # Resolve pathing
                t0 = time()
                overlap = self.agents.gt(0.5) & (self._other_agents() |
                                                 self.pathing.repeat_interleave(self.num_agents, 0) |
                                                 self.treasure.repeat_interleave(self.num_agents, 0))
                reset_due_to_pathing = overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
                if torch.any(reset_due_to_pathing):
                    self.agents[reset_due_to_pathing & currently_moving] = original_agents[reset_due_to_pathing & currently_moving]
                self._log(f'Agent pathing: {1000 * (time() - t0)}ms')

        t0 = time()
        # Update orientations
        self.orientations[(actions == self.rotate_right) | (actions == self.move_forward_and_turn_right)] += 1
        self.orientations[(actions == self.rotate_left) | (actions == self.move_forward_and_turn_left)] += 3
        self.orientations.fmod_(4)
        self._log(f'Orientations: {1000 * (time() - t0)}ms')

        self.has_dug = actions == self.dig
        if torch.any(self.has_dug):
            pass

        self.env_lifetimes += 1
        observations = self._observe()
        dones = {f'agent_{i}': d for i, d in
                 enumerate(self.dones.clone().view(self.num_envs, self.num_agents).t().unbind())}
        # # Environment is done if its past the maximum episode length
        dones['__all__'] = self.env_lifetimes.gt(self.max_env_lifetime)

        rewards = {f'agent_{i}': d for i, d in
                   enumerate(self.rewards.clone().view(self.num_envs, self.num_agents).t().unbind())}

        info.update({
            f'hp_{i}': d for i, d in
            enumerate(self.hp.clone().view(self.num_envs, self.num_agents).t().unbind())
        })
        for i_action in range(self.num_actions):
            action_i = actions == i_action
            info.update({
                f'action_{i_action}_{i_agent}': d for i_agent, d in
                enumerate(action_i.clone().view(self.num_envs, self.num_agents).t().unbind())
            })

        return observations, rewards, dones, info

    def reset(self, done: torch.Tensor = None, return_observations: bool = True) -> Optional[Dict[str, torch.Tensor]]:
        # Reset envs that contain no snakes
        if done is None:
            done = self.dones.view(self.num_envs, self.num_agents).all(dim=1)

        if self.colour_mode == 'random':
            # Change agent colours each death
            new_colours = self._get_n_colours(self.dones.sum().item())
            self.agent_colours[self.dones] = new_colours

        # Reset any environment which has passed the maximum lifetime. This will also regenerate the pathing
        # map if the pathing generator is randomised
        if torch.any(done):
            num_done = done.sum().item()
            agent_dones = done.repeat_interleave(self.num_agents)
            new_agents, new_orientations, new_dones, new_hp, new_pathing, new_respawns = self._create_envs(num_done)
            self.agents[agent_dones] = new_agents
            self.orientations[agent_dones] = new_orientations
            self.dones[agent_dones] = new_dones
            self.hp[agent_dones] = new_hp
            # If we are using a fixed MapGenerator these lines won't do anything
            self.pathing[done] = new_pathing
            self.respawns[done] = new_respawns

            self.env_lifetimes[done] = 0

        # Single agent resets
        if torch.any(self.dones):
            first_done_per_env = (self.dones.view(self.num_envs, self.num_agents).cumsum(
                dim=1) == 1).flatten() & self.dones

            pathing = self.pathing.clone()
            pathing |= self.agents\
                .view(self.num_envs, self.num_agents, self.height, self.width)\
                .gt(0.5)\
                .any(dim=1, keepdim=True)
            pathing = pathing.repeat_interleave(self.num_agents, 0)
            pathing = pathing[first_done_per_env]

            available_respawn_locations = self.respawns.repeat_interleave(self.num_agents, 0)
            available_respawn_locations = available_respawn_locations[first_done_per_env]

            new_agents, new_orientations, sucessfully_respawned = self._respawn(pathing, available_respawn_locations, False)

            self.agents[first_done_per_env] = new_agents
            self.orientations[first_done_per_env] = new_orientations
            self.hp[first_done_per_env] = self.initial_hp
            self.dones[first_done_per_env] = sucessfully_respawned

        if return_observations:
            return self._observe()

    def check_consistency(self):
        self.errors = torch.zeros_like(self.errors)

        # Only one agent per env
        agents_per_env = self.agents.view(self.num_envs, -1).sum(dim=1)
        if torch.any(agents_per_env.gt(self.num_agents)):
            bad_envs = agents_per_env.gt(self.num_agents)
            msg = 'Too many agents in {} envs.'.format(bad_envs.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Overlapping agents
        overlapping_agents = self.agents.view(self.num_envs, self.num_agents, self.height, self.width) \
            .sum(dim=1, keepdim=True) \
            .view(self.num_envs, -1) \
            .max(dim=-1)[0] \
            .gt(1.5)
        self.errors |= overlapping_agents
        if torch.any(overlapping_agents):
            msg = 'Agent-agent overlap in {} envs.'.format(overlapping_agents.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Pathing overlap
        agent_pathing_overlap = self.pathing.repeat_interleave(self.num_agents, 0) & self.agents.gt(0.5)
        if torch.any(agent_pathing_overlap):
            bad_envs = agent_pathing_overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
            msg = 'Agent-wall overlap in {} envs.'.format(bad_envs.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Treasure overlap
        print(self.treasure.device, self.agents.device)
        agent_treasure_overlap = self.treasure.repeat_interleave(self.num_agents, 0) & self.agents.gt(0.5)
        if torch.any(agent_treasure_overlap):
            bad_envs = agent_treasure_overlap.view(self.num_envs * self.num_agents, -1).any(dim=1)
            msg = 'Agent-treasure overlap in {} envs.'.format(bad_envs.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Can only get reward when digging

        # Max reward per step is 1
        impossible_rewards = self.rewards.gt(1 + EPS)
        # self.errors |= impossible_rewards
        if torch.any(impossible_rewards):
            msg = 'Agent has received impossibly high reward in {} envs'.format(impossible_rewards.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Every alive agent must actually exist
        alive = ~self.dones
        agent_exists = self.agents.view(self.num_envs*self.num_agents, -1).sum(dim=1).gt(0.5)
        alive_and_doesnt_exist = alive & (~agent_exists)
        # self.errors |= alive_and_doesnt_exist
        if torch.any(alive_and_doesnt_exist):
            msg = 'Agent is alive but doesn\'t exist in {} envs.'.format(alive_and_doesnt_exist.sum().item())
            if self.strict:
                raise RuntimeError(msg)
            else:
                warnings.warn(msg)

        # Reset bad envs. I choose not to reset the env lifetime so we can maintain the nice
        # thing of having all of the envs terminate synchronously.
        if torch.any(self.errors):
            num_done = self.errors.sum().item()
            agent_dones = self.errors.repeat_interleave(self.num_agents)
            new_agents, new_orientations, new_dones, new_hp, new_pathing, new_respawns = self._create_envs(num_done)
            self.agents[agent_dones] = new_agents
            self.orientations[agent_dones] = new_orientations
            self.dones[agent_dones] = new_dones
            self.hp[agent_dones] = new_hp
            # If we are using a fixed MapGenerator this line won't do anything
            self.pathing[self.errors] = new_pathing
            self.respawns[self.errors] = new_respawns

    def _respawn(self, pathing: torch.Tensor, respawns: torch.Tensor, exception_on_failure: bool):
        """Respawns an agent by picking randomly from allowed locations.

        Args:
            pathing: Pathing maps for the a batch of environments. This should be a Tensor of
                shape (n, 1, h, w) and dtype uint8.
            respawns: Maps of allowed respawn positions for a batch of environments. This should be a Tensor of
                shape (n, 1, h, w) and dtype uint8
            exception_on_failure:
        """
        n = pathing.shape[0]

        respawns &= ~pathing

        respawn_indices = drop_duplicates(torch.nonzero(respawns), 0)

        new_agents = torch.sparse_coo_tensor(
            respawn_indices.t(), torch.ones(len(respawn_indices)), respawns.shape,
            device=self.device, dtype=self.dtype
        ).to_dense()

        # Change orientation randomly
        new_orientations = torch.randint(0, 4, size=(n, ), device=self.device)

        dones = torch.zeros(n, dtype=torch.uint8, device=self.device)

        return new_agents, new_orientations, dones

    def _create_envs(self, num_envs: int) -> Tuple[torch.Tensor, ...]:
        """Gets pathing from pathing generator and repeatedly respawns agents until torch.all(dones == 0).

        Args:
            num_envs: Number of envs to create
        """
        dones = torch.ones(num_envs*self.num_agents, dtype=torch.uint8, device=self.device)

        map = self.map_generator.generate(num_envs)
        respawns = map.respawn
        pathing = map.pathing
        treasure = map.treasure

        agents = torch.zeros((num_envs * self.num_agents, 1, self.height, self.width), dtype=self.dtype,
                             device=self.device, requires_grad=False)
        orientations = torch.zeros((num_envs * self.num_agents), dtype=torch.long, device=self.device,
                                   requires_grad=False)
        hp = torch.ones(
            (num_envs * self.num_agents), dtype=torch.long, device=self.device, requires_grad=False) * self.initial_hp

        while torch.any(dones):
            first_done_per_env = (dones.view(num_envs, self.num_agents).cumsum(
                dim=1) == 1).flatten() & dones

            _pathing = pathing.clone().repeat_interleave(self.num_agents, 0)
            _pathing |= agents \
                .view(num_envs, self.num_agents, self.height, self.width) \
                .gt(0.5) \
                .any(dim=1, keepdim=True)\
                .repeat_interleave(self.num_agents, 0)
            _pathing = _pathing[first_done_per_env]

            available_respawn_locations = respawns.repeat_interleave(self.num_agents, 0)
            available_respawn_locations = available_respawn_locations[first_done_per_env]

            new_agents, new_orientations, sucessfully_respawned = self._respawn(_pathing, available_respawn_locations,
                                                                                False)

            agents[first_done_per_env] += new_agents
            orientations[first_done_per_env] = new_orientations
            hp[first_done_per_env] = self.initial_hp
            dones[first_done_per_env] = sucessfully_respawned

        return agents, orientations, dones, hp, pathing, respawns, treasure

    def _get_env_images(self) -> torch.Tensor:
        t0 = time()
        # Black background
        img = torch.zeros((self.num_envs, 3, self.height, self.width), device=self.device, dtype=torch.short)

        # Add slight highlight for orientation
        locations = (self.agents > 0.5).float()
        filters = ORIENTATION_FILTERS.to(dtype=self.dtype, device=self.device)
        orientation_highlights = F.conv2d(
            locations,
            filters,
            padding=1
        ) * 31
        directions_onehot = F.one_hot(self.orientations, 4).to(self.dtype)
        orientation_highlights = torch.einsum('bchw,bc->bhw', [orientation_highlights, directions_onehot]) \
            .view(self.num_envs, self.num_agents, 1, self.height, self.width) \
            .sum(dim=1) \
            .short()
        img += orientation_highlights.expand_as(img)

        # Add treasure
        per_env_lasers = self.treasure.reshape(self.num_envs, 1, self.height, self.width).sum(dim=1).gt(0.5)
        # Convert to NHWC axes for easier indexing here
        img = img.permute((0, 2, 3, 1))
        laser_colour = torch.tensor([127, 127, 31], device=self.device, dtype=torch.short)
        img[per_env_lasers] = laser_colour
        # Convert back to NCHW axes
        img = img.permute((0, 3, 1, 2))

        # Add colours for agents
        body_colours = torch.einsum('nhw,nc->nchw', [locations.squeeze(), self.agent_colours.float()]) \
            .view(self.num_envs, self.num_agents, 3, self.height, self.width) \
            .sum(dim=1) \
            .short()
        img += body_colours

        # Walls are grey
        img[self.pathing.expand_as(img)] = 127
        self._log(f'Rendering: {1000 * (time() - t0)}ms')

        return img

    def _observe(self) -> Dict[str, torch.Tensor]:
        t0 = time()
        observations = self.observation_fn.observe(self)
        self._log(f'Observations: {1000 * (time() - t0)}ms')
        return observations

    def _get_n_colours(self, n: int) -> torch.Tensor:
        colours = torch.rand((n, 3), device=self.device)
        colours /= colours.norm(2, dim=1, keepdim=True)
        colours *= 192
        colours = colours.short()
        return colours

    @property
    def x(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) // self.width

    @property
    def y(self):
        return self.agents.view(self.num_envs*self.num_agents, -1).argmax(dim=1) % self.width
