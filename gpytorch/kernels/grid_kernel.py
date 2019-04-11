#!/usr/bin/env python3

import torch
from .kernel import Kernel
from ..lazy import delazify, ToeplitzLazyTensor, KroneckerProductLazyTensor
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
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    def __init__(self, base_kernel, grid, interpolation_mode=False, active_dims=None):
        super(GridKernel, self).__init__(active_dims=active_dims)
        self.interpolation_mode = interpolation_mode
        self.base_kernel = base_kernel
        self.register_buffer("grid", grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        self.grid.detach().resize_(grid.size()).copy_(grid)

        if not self.interpolation_mode:
            full_grid = create_data_from_grid(self.grid)
            self.full_grid.detach().resize_(full_grid).copy_(full_grid)

        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        grid = self.grid

        if not self.interpolation_mode:
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat

            n_dim = x1.size(-1)

            if settings.use_toeplitz.on():
                first_item = grid[0:1]
                covar_columns = self.base_kernel(first_item, grid, diag=False, last_dim_is_batch=True, **params)
                covar_columns = delazify(covar_columns).squeeze(-2)
                if last_dim_is_batch:
                    covars = [ToeplitzLazyTensor(covar_columns.squeeze(-2))]
                else:
                    covars = [ToeplitzLazyTensor(covar_columns[i : i + 1].squeeze(-2)) for i in range(n_dim)]
            else:
                full_covar = self.base_kernel(grid, grid, last_dim_is_batch=True, **params)
                if last_dim_is_batch:
                    covars = [full_covar]
                else:
                    covars = [full_covar[i : i + 1].squeeze(0) for i in range(n_dim)]

            if len(covars) > 1:
                covar = KroneckerProductLazyTensor(*covars[::-1])
            else:
                covar = covars[0]

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            return self.base_kernel.forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
