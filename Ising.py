import numpy as np


def ising_energy(grid, J, h):
    left_shift = shift_left(grid)
    right_shift = shift_right(grid)
    up_shift = shift_up(grid)
    down_shift = shift_down(grid)
    # Divide by two because this double-counts the pairs of neighbors.
    interactions = 0.5 * np.sum(
        left_shift * grid + right_shift * grid + up_shift * grid + down_shift * grid
    )
    background = np.sum(grid)
    total_energy = -J * interaction - h * grid
    return total_energy


def grid_to_list(grid):
    return grid.flatten()


def list_to_grid(array):
    N = int(len(array) ** 0.5)
    return np.reshape(array, (N, N))


def shift_down(grid):
    N = grid.shape[0]
    last_row = np.reshape(grid[-1, :], (1, N))
    return np.append(last_row, grid[:-1, :], axis=0)


def shift_up(grid):
    N = grid.shape[0]
    first_row = np.reshape(grid[0, :], (1, N))
    return np.append(grid[1:, :], first_row, axis=0)


def shift_left(grid):
    N = grid.shape[0]
    first_col = np.reshape(grid[:, 0], (N, 1))
    return np.append(grid[:, 1:], first_col, axis=1)


def shift_right(grid):
    N = grid.shape[0]
    last_col = np.reshape(grid[:, -1], (N, 1))
    return np.append(last_col, grid[:, :-1], axis=1)
