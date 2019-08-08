import matplotlib.pyplot as plt
import numpy as np

from multigrid.envs.laser_tag.map_generators import generate_random_mazes, generate_random_respawns


# num_envs = 1024
num_envs = 25

# height = 9
# width = 9
# complexity = 0.025
# density = 0.5
# num_respawns = 4

# height = 9
# width = 16
# # complexity = 0.025
# complexity = 0.015
# # density = 0.5
# density = 0.6
# num_respawns = 4

height = 14
width = 22
# complexity = 0.025
complexity = 0.01
# density = 0.5
density = 0.4
num_respawns = 7

mazes = generate_random_mazes(num_envs, height, width, complexity, density, 'cuda')
respawns = generate_random_respawns(mazes, num_respawns, 2)

from config import PATH
np.save(f'{PATH}/data/pathing-small4.npy', mazes.cpu().numpy())
np.save(f'{PATH}/data/respawn-small4.npy', respawns.cpu().numpy())


# fig, axes = plt.subplots(np.ceil(np.sqrt(num_envs)).astype(int), np.ceil(np.sqrt(num_envs)).astype(int), figsize=(10, 10))
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

for i, m in enumerate(mazes[:25]):
    if isinstance(axes, np.ndarray):
        ax = axes.ravel()[i]
    else:
        ax = axes

    ax.imshow(
        mazes[i, 0].cpu().numpy() + respawns[i, 0].cpu().numpy()*0.5,
        cmap=plt.cm.binary,
        interpolation='nearest'
    )
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
