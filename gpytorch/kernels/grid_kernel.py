#!/usr/bin/env python3

from typing import List

import torch
from torch import Tensor

from .. import settings
from ..lazy import KroneckerProductLazyTensor, ToeplitzLazyTensor, cat, delazify
from ..utils.grid import convert_legacy_grid, create_data_from_grid
from .kernel import Kernel


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
        :attr:`grid` (Tensor):
            A g x d tensor where column i consists of the projections of the
            grid in dimension i.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
        :attr:`interpolation_mode` (bool):
            Used for GridInterpolationKernel where we want the covariance
            between points in the projections of the grid of each dimension.
            We do this by treating `grid` as d batches of g x 1 tensors by
            calling base_kernel(grid, grid) with last_dim_is_batch to get a d x g x g Tensor
            which we Kronecker product to get a g x g KroneckerProductLazyTensor.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    is_stationary = True

    def __init__(
        self, base_kernel: Kernel, grid: List[Tensor], interpolation_mode: bool = False, active_dims: bool = None
    ):
        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")

        super().__init__(active_dims=active_dims)
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)
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

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    @property
    def grid(self):
        return [getattr(self, f"grid_{i}") for i in range(self.num_dims)]

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)

        if len(grid) != self.num_dims:
            raise RuntimeError("New grid should have the same number of dimensions as before.")

        for i in range(self.num_dims):
            setattr(self, f"grid_{i}", grid[i])

        if not self.interpolation_mode:
            self.full_grid = create_data_from_grid(self.grid)

        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    @property
    def is_ragged(self):
        return not all(self.grid[0].size() == proj.size() for proj in self.grid)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        grid = self.grid
        if last_dim_is_batch and self.is_ragged:
            raise ValueError("last_dim_is_batch requires all dimensions to have same number of grid points")

        if not self.interpolation_mode:
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat
            # Can exploit Toeplitz structure if grid points in each dimension are equally
            # spaced and using a translation-invariant kernel
            if settings.use_toeplitz.on():
                first_grid_point = [proj[0].unsqueeze(0) for proj in grid]
                covars = [
                    self.base_kernel(first, proj, last_dim_is_batch=False, **params)
                    for first, proj in zip(first_grid_point, grid)
                ]  # Each entry i contains a 1 x grid_size[i] covariance matrix
                covars = [delazify(c) for c in covars]

                if last_dim_is_batch:
                    # Toeplitz expects batches of columns so we concatenate the
                    # 1 x grid_size[i] tensors together
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = ToeplitzLazyTensor(torch.cat(covars, dim=-2))
                else:
                    # Non-batched ToeplitzLazyTensor expects a 1D tensor, so we squeeze out the row dimension
                    covars = [ToeplitzLazyTensor(c.squeeze(-2)) for c in covars]
                    # Due to legacy reasons, KroneckerProductLazyTensor(A, B, C) is actually (C Kron B Kron A)
                    covar = KroneckerProductLazyTensor(*covars[::-1])
            else:
                covars = [
                    self.base_kernel(proj, proj, last_dim_is_batch=False, **params) for proj in grid
                ]  # Each entry i contains a grid_size[i] x grid_size[i] covariance matrix
                if last_dim_is_batch:
                    # Note that this requires all the dimensions to have the same number of grid points
                    covar = cat([c.unsqueeze(-3) for c in covars], dim=-3)
                else:
                    covar = KroneckerProductLazyTensor(*covars[::-1])

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            return self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
