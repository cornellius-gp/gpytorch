#!/usr/bin/env python3

import math
import torch


def scale_to_bounds(x, lower_bound, upper_bound):
    """
    Scale the input data so that it lies in between the lower and upper bounds.

    Args:
        :attr:`x` (Tensor `n` or `b x n`):
            the input
        :attr:`lower_bound` (float)
        :attr:`upper_bound` (float)

    Returns:
        :obj:`torch.Tensor`
    """
    # Scale features so they fit inside grid bounds
    min_val = x.min()
    max_val = x.max()
    diff = max_val - min_val
    x = (x - min_val) * (0.95 * (upper_bound - lower_bound) / diff) + 0.95 * lower_bound
    return x


def choose_grid_size(train_inputs, ratio=1.0):
    """
    Given some training inputs, determine a good grid size for KISS-GP.

    Args:
        :attr:`train_inputs` (Tensor `n` or `n x d` or `b x n x d`):
            training data
        :attr:`ratio` (float, optional):
            Ratio - number of grid points to the amount of data (default: 1.)

    Returns:
        :obj:`int`
    """
    # Scale features so they fit inside grid bounds
    num_data = train_inputs.numel() if train_inputs.dim() == 1 else train_inputs.size(-2)
    num_dim = 1 if train_inputs.dim() == 1 else train_inputs.size(-1)
    return int(ratio * math.pow(num_data, 1.0 / num_dim))


def create_data_from_grid(grid):
    grid_size = grid.size(-2)
    grid_dim = grid.size(-1)
    grid_data = torch.zeros(int(pow(grid_size, grid_dim)), grid_dim, device=grid.device)
    prev_points = None
    for i in range(grid_dim):
        for j in range(grid_size):
            grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[j, i])
            if prev_points is not None:
                grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
        prev_points = grid_data[: grid_size ** (i + 1), : (i + 1)]

    return grid_data
