#!/usr/bin/env python3

import torch
from typing import List
from torch import Tensor
from .kernel import Kernel
from ..lazy import ToeplitzLazyTensor, KroneckerProductLazyTensor
from .. import settings
from gpytorch.utils.grid import create_data_from_grid


class GridKernel(Kernel):
    r"""
    If the input data :math:`X` are regularly spaced on a grid, then
    `GridKernel` can dramatically speed up computatations for stationary kernel.

    GridKernel exploits Toeplitz and Kronecker structure within the covariance matrix.
    See `Fast kernel learning for multidimensional pattern extrapolation`_ for more info.

    .. note::

        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to speed up with grid methods.
        :attr:`grid` (List[Tensor]):
            Each element of the list contains the increments of the grid in that dimension
        :attr:`interpolation_mode`:
            Where we plan to do interpolation like in GridInterpolationKernel
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    def __init__(
        self, base_kernel: Kernel, grid: List[Tensor], interpolation_mode: bool = False, active_dims: bool = None
    ):
        super().__init__(active_dims=active_dims)
        self.interpolation_mode = interpolation_mode
        self.base_kernel = base_kernel
        self.num_dims = len(grid)
        self.register_buffer_list("grid", grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def register_buffer_list(self, base_name, tensors):
        """Helper to register several buffers at once under a single base name"""
        for i, tensor in enumerate(tensors):
            self.register_buffer(base_name + "_" + str(i), tensor)

    @property
    def grid(self):
        return [getattr(self, "grid" + "_" + str(i)) for i in range(self.num_dims)]

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super().train(mode)

    def update_grid(self, grid: List[Tensor]):
        """
        Supply a new `grid` if it ever changes.
        """
        if not isinstance(grid, list):
            raise RuntimeError("Update_grid requires that grid is a list of tensors of grid points along each axis.")
        if len(grid) != self.num_dims:
            raise RuntimeError("New grid should have the same number of dimensions as before.")

        for dim in range(self.num_dims):
            self.grid[dim].detach_().resize_(grid[dim].size()).copy_(grid[dim])

        if not self.interpolation_mode:
            full_grid = create_data_from_grid(self.grid)
            self.full_grid.detach_().resize_(full_grid).copy_(full_grid)

        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params):
        grid = self.grid

        if not self.interpolation_mode:  # TODO: Update based on possible jagged grid shapes?
            # Use the same grid for all batches
            full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat
            if settings.use_toeplitz.on():
                first_item = [proj[0:1] for proj in grid]  # Each entry is torch.Size([1])
                covar_columns = [
                    self.base_kernel(first, proj, last_dim_is_batch=False, **params)
                    for first, proj in zip(first_item, grid)
                ]  # Now each entry i is of size 1 x grid_size[i]

                if last_dim_is_batch:
                    # For b x n x d input, we want a b x d x n x n Toeplitz matrix
                    # Toeplitz expects batches of columns so we first squeeze out the row dimension
                    # Then we stack them together
                    # TODO:
                    covar = [ToeplitzLazyTensor(c.squeeze(-2)) for c in covar_columns]
                else:
                    # Toeplitz expects batches of columns so we first squeeze out the row dimension
                    covar = [ToeplitzLazyTensor(c.squeeze(-2)) for c in covar_columns]
            else:
                full_covar = [self.base_kernel(proj, proj, last_dim_is_batch=False, **params) for proj in grid]
                if last_dim_is_batch:
                    # TODO:
                    covar = full_covar
                else:
                    covar = full_covar
            if len(covar) > 1:
                # Need to reverse covars to make KroneckerProductLazyTensor matmul properly
                covar = KroneckerProductLazyTensor(*covar[::-1])
            else:
                covar = covar[0]

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            return self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
