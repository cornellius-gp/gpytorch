#!/usr/bin/env python3

import math
import torch
from typing import List
from functools import reduce
from operator import mul


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


def choose_grid_size(train_inputs, ratio=1.0, kronecker_structure=True):
    """
    Given some training inputs, determine a good grid size for KISS-GP.

    Args:
        :attr:`train_inputs` (Tensor `n` or `n x d` or `b x n x d`):
            training data
        :attr:`ratio` (float, optional):
            Ratio - number of grid points to the amount of data (default: 1.)
        :attr:`kronecker_structure` (bool, default=True):
            Whether or not the model will use Kronecker structure in the grid
            (set to True unless there is an additive or product decomposition in the prior)

    Returns:
        :obj:`int`
    """
    # Scale features so they fit inside grid bounds
    num_data = train_inputs.numel() if train_inputs.dim() == 1 else train_inputs.size(-2)
    num_dim = 1 if train_inputs.dim() == 1 else train_inputs.size(-1)
    if kronecker_structure:
        return int(ratio * math.pow(num_data, 1.0 / num_dim))
    else:
        return (ratio * num_data)


def create_data_from_grid(grid: List[torch.Tensor]):
    # grid_size = grid.size(-2)
    grid_dim = len(grid)
    grid_sizes = [grid[i].size(0) for i in range(grid_dim)]
    grid_data = torch.zeros(reduce(mul, grid_sizes), grid_dim, device=grid[0].device)
    prev_points = None
    num_pts_i = 1
    for i in range(grid_dim):  # fill the values of dimension i
        grid_size = grid_sizes[i]
        for j in range(grid_size):
            grid_data[j * num_pts_i: (j + 1) * num_pts_i, i].fill_(grid[i][j])
            if prev_points is not None:
                grid_data[j * num_pts_i: (j + 1) * num_pts_i, :i].copy_(prev_points)
        num_pts_i *= grid_size
        prev_points = grid_data[: num_pts_i, : (i + 1)]

    return grid_data
