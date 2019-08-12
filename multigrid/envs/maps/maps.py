from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Callable
import numpy as np
import torch

from multigrid.envs.laser_tag.maps import MAPS as LASER_TAG_MAPS
from multigrid.envs.treasure_hunt.maps import MAPS as TREASURE_HUNT_MAPS
from multigrid.envs.maps.procedural import generate_random_mazes, generate_random_respawns


handcrafted_maps = {
    'laser_tag': LASER_TAG_MAPS,
    'treasure_hunt': TREASURE_HUNT_MAPS
}


class MapBatch:
    """A map must have a pathing and respawn map and may have other tensors."""
    def __init__(self, pathing: torch.Tensor, respawn: torch.Tensor, **other_tensors: torch.Tensor):
        self.pathing = pathing
        self.respawn = respawn
        self.other_tensors = {}
        self.other_tensors.update(other_tensors)

    def __getattr__(self, item) -> torch.Tensor:
        if item in ('pathing', 'respawn'):
            return getattr(self, item)
        else:
            return self.other_tensors[item]


def parse_mapstring(mapstring: Union[str, List[str]], map_symbols: Optional[Dict[str, str]] = None) -> MapBatch:
    if not isinstance(mapstring, list):
        # The name of a pre-specified map has been passed
        env = mapstring.split('-')[0]
        map_name = mapstring.split('-')[1]
        mapstring = handcrafted_maps[env][map_name]

    if map_symbols is None:
        map_symbols = {'*': 'pathing', 'P': 'respawn', 'T': 'treasure'}

    # Get height and width
    height = len(mapstring)
    width = (len(mapstring[0]) + 1) // 2

    map_tensors = {name: torch.zeros((1, 1, height, width), dtype=torch.uint8) for name in map_symbols.values()}
    for i, line in enumerate(mapstring):
        # Remove padding spaces
        line = (line + ' ')[::2]

        if len(line) != width:
            raise ValueError('Map string has inconsistent shape')

        for symbol, name in map_symbols.items():
            tensor_row = torch.tensor([char == symbol for char in line])
            map_tensors[name][:, :, i, :] = tensor_row

    return MapBatch(**map_tensors)


class MapGenerator(ABC):
    """Base class for map generators."""
    @abstractmethod
    def generate(self, num_envs: int) -> MapBatch:
        raise NotImplementedError


class FixedMapGenerator(MapGenerator):
    def __init__(self, maps: MapBatch, device: str):
        self.maps = maps
        self.device = device

    def generate(self, num_envs: int) -> MapBatch:
        pathing = self.maps.pathing.to(self.device).repeat((num_envs, 1, 1, 1))
        respawn = self.maps.respawn.to(self.device).repeat((num_envs, 1, 1, 1))
        other_tensors = {k: v.to(self.device).repeat((num_envs, 1, 1, 1)) for k, v in self.maps.other_tensors.items()}
        return MapBatch(pathing, respawn, **other_tensors)

    @property
    def height(self):
        return self.maps.pathing.size(2)

    @property
    def width(self):
        return self.maps.pathing.size(3)


def concatenate(map_batches: List[MapBatch]) -> MapBatch:
    pathing = torch.cat([m.pathing for m in map_batches], dim=0)
    respawn = torch.cat([m.respawn for m in map_batches], dim=0)
    other_tensors = {
        k: torch.cat([m.other_tensors[k] for m in map_batches], dim=0)
        for k in map_batches[0].other_tensors.keys()
    }

    return MapBatch(pathing, respawn, **other_tensors)


class MapPool(MapGenerator):
    """Uniformly selects maps at random from a pool of fixed maps."""
    def __init__(self, maps: Union[MapBatch, List[MapBatch]]):
        if isinstance(maps, list):
            # Concatenate
            maps = concatenate(maps)

        self.maps = maps
        self.num_maps = self.maps.pathing.size(0)

    def generate(self, num_envs: int) -> MapBatch:
        indices = torch.randint(low=0, high=self.num_maps, size=(num_envs,))
        pathing = self.maps.pathing[indices]
        respawn = self.maps.respawn[indices]
        other_tensors = {k: v[indices] for k, v in self.maps.other_tensors}
        return MapBatch(pathing, respawn, **other_tensors)


class Random(MapGenerator):
    """Generates a new MapBatch each time generate is called."""
    def __init__(self, num_respawns: int,
                 height: int,
                 width: int,
                 maze_complexity: float,
                 maze_density: float,
                 device: str,
                 min_respawn_separation: int = 2,
                 other_tensor_generator: Optional[Callable] = None):
        self.num_respawns = num_respawns
        self.height = height
        self.width = width
        self.maze_complexity = maze_complexity
        self.maze_density = maze_density
        self.device = device
        self.min_respawn_separation = min_respawn_separation
        self.other_tensor_generator = other_tensor_generator

    def generate(self, num_envs: int) -> MapBatch:
        pathing = generate_random_mazes(num_envs, self.height, self.width, self.maze_complexity, self.maze_density,
                                        self.device)
        respawn = generate_random_respawns(pathing, self.num_respawns, self.min_respawn_separation)
        if self.other_tensor_generator is None:
            other_tensors = {}
        else:
            other_tensors = self.other_tensor_generator(pathing, respawn)

        return MapBatch(pathing, respawn, **other_tensors)


def maps_from_file(pathing: str,
                   respawn: str,
                   device: str,
                   num_maps: Optional[int] = None,
                   **other_tensors) -> MapBatch:
    pathing = torch.from_numpy(np.load(pathing)).to(dtype=torch.uint8, device=device)
    respawn = torch.from_numpy(np.load(respawn)).to(dtype=torch.uint8, device=device)
    other_tensors = {
        k: torch.from_numpy(np.load(v)).to(dtype=torch.uint8, device=device) for k, v in other_tensors.items()}

    if num_maps is not None:
        if pathing.size(0) < num_maps or respawn.size(0) < num_maps:
            raise ValueError('Not enough maps in file to meet `num_maps` argument.')

        pathing, respawn = pathing[:num_maps], respawn[:num_maps]
        other_tensors = {k: v[:num_maps] for k, v in other_tensors}

    if pathing.size() != respawn.size():
        raise ValueError('Incompatible pathing and respawn shapes.')

    return MapBatch(pathing, respawn, **other_tensors)

