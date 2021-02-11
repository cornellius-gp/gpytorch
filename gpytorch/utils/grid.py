#!/usr/bin/env python3

import math
from typing import List, Tuple

import torch


def scale_to_bounds(x, lower_bound, upper_bound):
    """
    Scale the input data so that it lies in between the lower and upper bounds.

    :param x: the input data
    :type x: torch.Tensor (... x n x d)
    :param float lower_bound: lower bound of scaled data
    :param float upper_bound: upper bound of scaled data
    :return: scaled data
    :rtype: torch.Tensor (... x n x d)
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

    :param x: the input data
    :type x: torch.Tensor (... x n x d)
    :param ratio: Amount of grid points per data point (default: 1.)
    :type ratio: float, optional
    :param kronecker_structure: Whether or not the model will use Kronecker structure in the grid
        (set to True unless there is an additive or product decomposition in the prior)
    :type kronecker_structure: bool, optional
    :return: Grid size
    :rtype: int
    """
    # Scale features so they fit inside grid bounds
    num_data = train_inputs.numel() if train_inputs.dim() == 1 else train_inputs.size(-2)
    num_dim = 1 if train_inputs.dim() == 1 else train_inputs.size(-1)
    if kronecker_structure:
        return int(ratio * math.pow(num_data, 1.0 / num_dim))
    else:
        return ratio * num_data


def convert_legacy_grid(grid: torch.Tensor) -> List[torch.Tensor]:
    return [grid[:, i] for i in range(grid.size(-1))]


def create_data_from_grid(grid: List[torch.Tensor]) -> torch.Tensor:
    """
    :param grid: Each Tensor is a 1D set of increments for the grid in that dimension
    :type grid: List[torch.Tensor]
    :return: The set of points on the grid going by column-major order
    :rtype: torch.Tensor
    """
    if torch.is_tensor(grid):
        grid = convert_legacy_grid(grid)
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


def create_grid(
    grid_sizes: List[int], grid_bounds: List[Tuple[float, float]], extend: bool = True, device="cpu", dtype=torch.float,
) -> List[torch.Tensor]:
    """
    Creates a grid represented by a list of 1D Tensors representing the
    projections of the grid into each dimension

    If `extend`, we extend the grid by two points past the specified boundary
    which can be important for getting good grid interpolations.

    :param grid_sizes: Sizes of each grid dimension
    :type grid_sizes: List[int]
    :param grid_bounds: Lower and upper bounds of each grid dimension
    :type grid_sizes: List[Tuple[float, float]]
    :param device: target device for output (default: cpu)
    :type device: torch.device, optional
    :param dtype: target dtype for output (default: torch.float)
    :type dtype: torch.dtype, optional
    :return: Grid points for each dimension. Grid points are stored in a :obj:`torch.Tensor` with shape `grid_sizes[i]`.
    :rtype: List[torch.Tensor]
    """
    grid = []
    for i in range(len(grid_bounds)):
        grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_sizes[i] - 2)
        if extend:
            proj = torch.linspace(
                grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_sizes[i], device=device, dtype=dtype,
            )
        else:
            proj = torch.linspace(grid_bounds[i][0], grid_bounds[i][1], grid_sizes[i], device=device, dtype=dtype,)
        grid.append(proj)
    return grid
