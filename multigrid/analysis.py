"""Functions for analysing experiment results"""
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
import os


colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
argtypes = {
    'size': int,
    'n_envs': int,
    'n_agents': int,
    'n_species': int,
    'lr': float,
    'gamma': float,
    'update_steps': int,
    'entropy': float,
    'norm_returns': lambda x: x.lower()[0] == 't',
    'gae_lambda': float,
    'share_backbone': lambda x: x.lower()[0] == 't',
    'r': int,
    'repeat': int
}


def parse(argstring: str) -> dict:
    args = {i.split('=')[0]: i.split('=')[1] for i in argstring[:-4].split('__')}
    return args


def get_all_repeats(experiment_folder: str, read_csv_kwargs={}) -> pd.DataFrame:
    """Combines the logs of all runs of an experiment into one DataFrame."""
    df = []
    root = f'{experiment_folder}/logs/'
    for base, _, files in os.walk(root):
        if base == root:
            for f in files:
                args = parse(f)
                _df = pd.read_csv(base+f, comment='#', **read_csv_kwargs)

                for arg, val in args.items():
                    _df[arg] = argtypes.get(arg, str)(val)

                df.append(_df)

    df = pd.concat(df, sort=False)
    return df


def block_diagonal(n_blocks: int, block_size: int) -> np.ndarray:
    """Generates a 2D block diagonal matrix."""
    num_pool = block_size
    n = n_blocks
    return block_diag(*[np.ones((num_pool, num_pool)) for _ in range(n//num_pool)])


def build_jpc_matrix(eval_folder: str,
                     n: int,
                     excluded_runs: List[int] = [],
                     read_csv_kwargs: dict = {},
                     metric: str = 'reward') -> np.ndarray:
    jpc_matrix = np.zeros((n, n, 2))
    for root, _, files in os.walk(eval_folder):
        for f in sorted(files):
            _df = pd.read_csv(root + f, comment='#', **read_csv_kwargs)
            i = int(f[:-4].split('-')[0])
            j = int(f[:-4].split('-')[1])

            jpc_matrix[i, j, 0] = _df[f'{metric}_0'].sum()
            jpc_matrix[i, j, 1] = _df[f'{metric}_1'].sum()

    included_runs = [i for i in range(n) if i not in excluded_runs]
    jpc_matrix = jpc_matrix[included_runs][:, included_runs]

    return jpc_matrix


def calculate_reward_metrics(jpc_matrix: np.ndarray, n_pool: int = 1) -> (float, float, float):
    j = jpc_matrix.sum(axis=2)
    if n_pool == 1:
        D = j.diagonal().mean()
        O = j[np.where(~np.eye(jpc_matrix.shape[0], dtype=bool))].mean()
    elif n_pool > 1:
        blocks = block_diagonal(jpc_matrix.shape[0], n_pool).astype(bool)
        D = j[blocks].mean()
        O = j[~blocks].mean()
    else:
        raise ValueError('n_pool must be >= 1')

    R = (D - O) / D

    return D, O, R


def parse_jpc_folder(jpc_folder: str,
                     n: int,
                     n_pool: int,
                     excluded_runs=[],
                     read_csv_kwargs={'nrows': 1000},
                     metric='reward') -> (np.ndarray, float, float, float):
    jpc_matrix = build_jpc_matrix(jpc_folder, n, excluded_runs, read_csv_kwargs, metric)
    D, O, R = calculate_reward_metrics(jpc_matrix, n_pool)
    return jpc_matrix, D, O, R


def plot_matchups(df: pd.DataFrame,
                  repeat: int,
                  n_pool: int,
                  alpha: float = 0.005,
                  metric: str = 'reward',
                  figsize: (float, float) = (16, 16)):
    """Plots training curves for all matchups in an agent pool run"""
    fig, axes = plt.subplots(n_pool, n_pool, figsize=figsize)
    for i in range(n_pool):
        for j in range(n_pool):
            if isinstance(axes, np.ndarray):
                ax = axes[i, j]
            else:
                ax = axes

            df_plot = df[(df[f'agent_id_0'] == i) & (df[f'agent_id_1'] == j) & (df['repeat'] == repeat)]

            ax.set_title('{}-{} vs {}-{}'.format(0, i, 1, j))
            ax.plot(df_plot[f'{metric}_0'].ewm(alpha=alpha).mean().values, color=colours[i])
            ax.plot(df_plot[f'{metric}_1'].ewm(alpha=alpha).mean().values, color=colours[n_pool + j])

            ax.set_xlim(left=0)
            ax.grid()

    return fig, axes
