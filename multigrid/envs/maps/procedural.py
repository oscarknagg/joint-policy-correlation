import torch
from torch.nn import functional as F

from config import EPS
from multigrid.utils import drop_duplicates


def generate_random_mazes(n: int, height: int, width: int, complexity: float, density: float, device: str) -> torch.Tensor:
    """Generates n random mazes where each empty point is guaranteed to be reachable from every other.

    This code is shamelessly ripped from the python code example from here:
    https://en.wikipedia.org/wiki/Maze_generation_algorithm
    which is a variant of Prim's algorithm. My only modification is to vectorise this across
    a batch dimension to generate many mazes in parallel.

    This functions contains a dirty hack to make sure mazes with even side
    lengths still have good pathing.

    Args:
        n: Number of mazes to generate
        height: Height of maze
        width: Width of maze
        complexity: Increases the sizes of wall-pieces
        density: Increases the number of wall-pieces
        device: Device on which the maze resides e.g. 'cuda', 'cpu'
    """
    complexity = int(complexity * (5 * (height + width)))  # number of components
    density = int(density * ((height // 2) * (width // 2)))  # size of components

    even_height = int((height % 2) == 0)
    even_width = int((width % 2) == 0)

    pathing = torch.zeros((n, 1, height, width), dtype=torch.uint8, device=device)
    pathing[:, :, 0, :] = 1
    pathing[:, :, -1, :] = 1
    pathing[:, :, :, 0] = 1
    pathing[:, :, :, -1] = 1

    for i in range(density):
        x = (torch.randint(0, (width - 3*even_width) // 2, size=(n,))) * 2
        y = (torch.randint(0, (height - 3*even_height) // 2, size=(n,))) * 2

        pathing[torch.arange(n), :, y, x] = 1

        for j in range(complexity):
            neighbours = torch.zeros((n, 4, 2))
            neighbours[:, 0] = torch.stack([x, y + 2]).t()
            neighbours[:, 1] = torch.stack([x, y - 2]).t()
            neighbours[:, 2] = torch.stack([x + 2, y]).t()
            neighbours[:, 3] = torch.stack([x - 2, y]).t()

            # Ignores neighbours off the edge
            clamped_neighbours = torch.stack([
                neighbours[:, :, 0].clamp(0, width - 1 - 2*even_width),
                neighbours[:, :, 1].clamp(0, height - 1 - 2*even_height),
            ], dim=-1)
            valid_neighbours = (neighbours == clamped_neighbours).all(dim=-1)

            # Pick random neighbours to be new x, y positions in each env
            tmp = drop_duplicates(torch.nonzero(valid_neighbours), 0)[:, 1]
            x_, y_ = neighbours[torch.arange(n), tmp].long().unbind(dim=1)

            # If new position is empty fill that and the intermediate position
            empty = pathing[torch.arange(n), :, y_, x_] == 0
            intermediate_x = (x + x_) // 2
            intermediate_y = (y + y_) // 2

            tmp_pathing = pathing[empty]

            n_empty = empty.sum().item()
            tmp_pathing[torch.arange(n_empty), intermediate_y[empty.squeeze()], intermediate_x[empty.squeeze()]] = 1
            tmp_pathing[torch.arange(n_empty), y_[empty.squeeze()], x_[empty.squeeze()]] = 1

            pathing[empty] = tmp_pathing

            x = x_
            y = y_

    return pathing


def generate_random_respawns(pathing: torch.Tensor, n: int, minimum_separation: int = 0) -> torch.Tensor:
    """Generates `n` respawn locations for each pathing map given as input.

    Args:
        pathing: Input pathing maps of shape (num_envs, 1, height, width)
        n: Number of respawn locations per pathing map
        minimum_separation: Minimum distance between respawn locations. Setting this too
            high for a particular map size might result having less than n respawn locations.
    """
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
