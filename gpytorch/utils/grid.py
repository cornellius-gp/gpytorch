#!/usr/bin/env python3

import math
import torch
from typing import List


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
        return ratio * num_data


def create_data_from_grid(grid: List[torch.Tensor]) -> torch.Tensor:
    """
    Args:
        :attr:`grid` (List[Tensor])
            Each Tensor is a 1D set of increments for the grid in that dimension
    Returns:
        `grid_data` (Tensor)
            Returns the set of points on the grid going by column-major order
            (due to legacy reasons).
    """
    ndims = len(grid)
    assert all(axis.dim() == 1 for axis in grid)
    projections = torch.meshgrid(*grid)
    grid_tensor = torch.stack(projections, axis=-1)
    # Note that if we did
    #     grid_data = grid_tensor.reshape(-1, ndims)
    # instead, we would be iterating through the points of our grid from the
    # last data dimension to the first data dimension. However, due to legacy
    # reasons, we need to iterate from the first data dimension to the last data
    # dimension when creating grid_data
    grid_data = grid_tensor.permute(*(reversed(range(ndims + 1)))).reshape(ndims, -1).transpose(0, 1)
    return grid_data
