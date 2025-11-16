import numpy as np


def neighbor_sum(grid):
    sum = np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1)
    return sum


def ising_energy(grid, J, h):
    interaction = np.sum(grid * neighbor_sum(grid)) / 2  # each pair counted twice
    background = np.sum(grid)
    total_energy = -J * interaction - h * background
    return total_energy


def grid_to_list(grid):
    return grid.flatten()


def list_to_grid(array):
    N = int(len(array) ** 0.5)
    return np.reshape(array, (N, N))


def magnetization(grid):
    return np.sum(grid) / grid.size

def energy_per_spin(grid, J, h):
    return ising_energy(grid, J, h) / grid.size


# def shift_down(grid):
#     N = grid.shape[0]
#     last_row = np.reshape(grid[-1, :], (1, N))
#     return np.append(last_row, grid[:-1, :], axis=0)


# def shift_up(grid):
#     N = grid.shape[0]
#     first_row = np.reshape(grid[0, :], (1, N))
#     return np.append(grid[1:, :], first_row, axis=0)


# def shift_left(grid):
#     N = grid.shape[0]
#     first_col = np.reshape(grid[:, 0], (N, 1))
#     return np.append(grid[:, 1:], first_col, axis=1)


# def shift_right(grid):
#     N = grid.shape[0]
#     last_col = np.reshape(grid[:, -1], (N, 1))
#     return np.append(last_col, grid[:, :-1], axis=1)
