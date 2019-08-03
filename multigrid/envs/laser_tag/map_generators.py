from abc import ABC, abstractmethod
from typing import List, NamedTuple, Union, Optional
import torch
import torch.nn.functional as F

from .maps import MAPS
from multigrid.utils import drop_duplicates
from config import EPS


def parse_mapstring(mapstring: List[str]) -> (torch.Tensor, torch.Tensor):
    # Get height and width
    height = len(mapstring)
    width = (len(mapstring[0]) + 1) // 2

    # Check consistent height and width
    # Convert to tensor
    pathing = torch.zeros((1, 1, height, width), dtype=torch.uint8)
    respawn = torch.zeros((1, 1, height, width), dtype=torch.uint8)
    for i, line in enumerate(mapstring):
        # Remove padding spaces
        line = (line + ' ')[::2]

        if len(line) != width:
            raise ValueError('Map string has inconsistent shape')

        _pathing = torch.tensor([char == '*' for char in line])
        pathing[:, :, i, :] = _pathing
        _respawn = torch.tensor([char == 'P' for char in line])
        respawn[:, :, i, :] = _respawn

    return pathing, respawn


class LaserTagMap(NamedTuple):
    pathing: torch.Tensor
    respawn: torch.Tensor


class LaserTagMapGenerator(ABC):
    """Base class for map generators."""
    @abstractmethod
    def generate(self, num_envs: int) -> LaserTagMap:
        raise NotImplementedError


class FixedMapGenerator(LaserTagMapGenerator):
    _pathing: torch.Tensor
    _respawn: torch.Tensor

    def __init__(self, device: str):
        self.device = device

    def generate(self, num_envs: int) -> LaserTagMap:
        pathing = self._pathing.to(self.device).repeat((num_envs, 1, 1, 1))
        respawn = self._respawn.to(self.device).repeat((num_envs, 1, 1, 1))
        return LaserTagMap(pathing, respawn)

    @property
    def height(self):
        return self._pathing.size(2)

    @property
    def width(self):
        return self._pathing.size(3)


class MapFromString(FixedMapGenerator):
    def __init__(self, mapstring: Union[str, List[str]], device: str):
        super(MapFromString, self).__init__(device)
        if isinstance(mapstring, list):
            # A mapstring has been passed
            self._pathing, self._respawn = parse_mapstring(mapstring)
        else:
            # The name of a pre-specified map has been passed
            mapstring = MAPS[mapstring]
            self._pathing, self._respawn = parse_mapstring(mapstring)


class MapPool(LaserTagMapGenerator):
    """Uniformly selects maps at random from a pool of fixed maps."""
    def __init__(self, map_pool: List[FixedMapGenerator]):
        assert len(map_pool) > 0
        self.map_pool = map_pool
        self.device = self.map_pool[0].device
        self.height = self.map_pool[0].height
        self.width = self.map_pool[0].width

    def generate(self, num_envs: int) -> LaserTagMap:
        map_selection = torch.randint(0, len(self.map_pool), size=(num_envs, ))

        pathing = torch.zeros((num_envs, 1, self.height, self.width), dtype=torch.uint8, device=self.device)
        respawn = torch.zeros((num_envs, 1, self.height, self.width), dtype=torch.uint8, device=self.device)

        for i in range(len(self.map_pool)):
            map_i = map_selection == i
            num_map_i = map_i.sum().item()
            new_maps = self.map_pool[i].generate(num_map_i)
            pathing[map_i] = new_maps.pathing
            respawn[map_i] = new_maps.respawn

        return LaserTagMap(pathing, respawn)


def generate_random_mazes(n: int, height: int, width: int, complexity: float, density: float, device: str):
    """Generates n random mazes that are guaranteed to have good pathing."""
    if not ((height % 2) == 1 and (width % 2) == 1):
        raise ValueError('Only odd sizes')

    complexity = int(complexity * (5 * (height + width)))  # number of components
    density = int(density * ((height // 2) * (width // 2)))  # size of components

    pathing = torch.zeros((n, 1, height, width), dtype=torch.uint8, device=device)
    pathing[:, :, 0, :] = 1
    pathing[:, :, -1, :] = 1
    pathing[:, :, :, 0] = 1
    pathing[:, :, :, -1] = 1

    for i in range(density):
        x = torch.randint(0, width // 2, size=(n,)) * 2
        y = torch.randint(0, height // 2, size=(n,)) * 2

        pathing[torch.arange(n), :, x, y] = 1

        for j in range(complexity):
            neighbours = torch.zeros((n, 4, 2))
            neighbours[:, 0] = torch.stack([x, y + 2]).t()
            neighbours[:, 1] = torch.stack([x, y - 2]).t()
            neighbours[:, 2] = torch.stack([x + 2, y]).t()
            neighbours[:, 3] = torch.stack([x - 2, y]).t()

            # Ignores neighbours off the edge
            clamped_neighbours = torch.stack([
                neighbours[:, :, 0].clamp(0, width),
                neighbours[:, :, 1].clamp(0, height),
            ], dim=-1)
            valid_neighbours = (neighbours == clamped_neighbours).all(dim=-1)

            # Pick random neighbours to be new x, y positions in each env
            tmp = drop_duplicates(torch.nonzero(valid_neighbours), 0)[:, 1]
            x_, y_ = neighbours[torch.arange(n), tmp].long().unbind(dim=1)

            # If new position is empty, fill it and the intermediate
            # position
            empty = pathing[torch.arange(n), :, x_, y_] == 0
            intermediate_x = (x + x_) // 2
            intermediate_y = (y + y_) // 2

            tmp_pathing = pathing[empty]

            n_empty = empty.sum().item()
            tmp_pathing[torch.arange(n_empty), intermediate_x[empty.squeeze()], intermediate_y[empty.squeeze()]] = 1
            tmp_pathing[torch.arange(n_empty), x_[empty.squeeze()], y_[empty.squeeze()]] = 1

            pathing[empty] = tmp_pathing

            x = x_
            y = y_

    return pathing


def generate_random_respawns(pathing: torch.Tensor, n: int, minimum_separation: int = 0):
    """Generates `n` respawn locations for each pathing map."""
    respawns = torch.zeros_like(pathing)

    for i in range(n):
        expanded_respawns = respawns.clone()
        if minimum_separation > 0:
            separation_filter = torch.ones((1, 1, 2 * minimum_separation + 1, 2 * minimum_separation + 1),
                                           device=respawns.device)

            expanded_respawns = F.conv2d(
                expanded_respawns.float(),
                separation_filter,
                padding=minimum_separation
            ).gt(EPS).byte()

        # Get empty locations
        available_locations = (pathing + expanded_respawns).sum(dim=1, keepdim=True).eq(0)

        respawn_locations = drop_duplicates(torch.nonzero(available_locations), 0)
        respawn_mask = torch.sparse_coo_tensor(
            respawn_locations.t(),
            torch.ones(len(respawn_locations)),
            available_locations.shape,
            device=pathing.device,
            dtype=pathing.dtype
        )
        respawn_mask = respawn_mask.to_dense()

        respawns += respawn_mask

    return respawns


class Random(LaserTagMapGenerator):
    def __init__(self, num_maps: int,
                 num_respawns: int,
                 height: int,
                 width: int,
                 maze_complexity: float,
                 maze_density: float,
                 device: str,
                 min_respawn_separation: int = 2):
        self.num_maps = num_maps
        self.pathing = generate_random_mazes(num_maps, height, width, maze_complexity, maze_density, device)
        self.respawn = generate_random_respawns(self.pathing, num_respawns, min_respawn_separation)

    def generate(self, num_envs: int) -> LaserTagMap:
        indices = torch.randint(low=0, high=self.num_maps, size=(num_envs, ))
        pathing = self.pathing[indices]
        respawn = self.respawn[indices]
        return LaserTagMap(pathing, respawn)
