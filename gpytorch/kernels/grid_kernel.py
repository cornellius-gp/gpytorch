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
        self.num_dims = len(grid)
        self.register_buffer_list('grid', grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def register_buffer_list(self, base_name, tensors):
        """Helper to register several buffers at once under a single base name"""
        for i, tensor in enumerate(tensors):
            self.register_buffer(base_name + '_' + str(i), tensor)

    @property
    def grid(self):
        return [getattr(self, 'grid' + '_' + str(i)) for i in range(self.num_dims)]


    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        if not isinstance(grid, list):
            raise RuntimeError("Update_grid requires that grid is a list of "
                               "tensors of grid points along each axis.")
        for dim in range(len(self.grid)):
            self.grid[dim].detach().resize_(grid[dim].size()).copy_(grid[dim])

        if not self.interpolation_mode:
            full_grid = create_data_from_grid(self.grid)
            self.full_grid.detach().resize_(full_grid).copy_(full_grid)

        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        grid = self.grid

        if not self.interpolation_mode:  # TODO: Update based on possible jagged grid shapes?
            if len(x1.shape[:-2]):
                full_grid = self.full_grid.expand(*x1.shape[:-2], *self.full_grid.shape[-2:])
            else:
                full_grid = self.full_grid

        if self.interpolation_mode or (torch.equal(x1, full_grid) and torch.equal(x2, full_grid)):
            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat

            if isinstance(x1, list):
                # we assume it is provided as a 'jagged tensor' representing a grid.
                n_dim = len(x1)
            else:
                n_dim = x1.size(-1)

            if settings.use_toeplitz.on():
                first_item = [self.grid[i][0:1] for i in range(len(self.grid))] # n_dim x 1
                # Instead of using last_dim_is_batch, use iterations... b/c "last dim" varies for each input dimension.
                # covar_columns = self.base_kernel(first_item, grid, diag=False, last_dim_is_batch=True, **params)
                # Also, like, how do I get this to work when last_dim_is_batch=True???
                covar_columns = [self.base_kernel(first_item[i], grid[i], last_dim_is_batch=False, **params) for i in
                                 range(n_dim)] # n_dim x 1 x grid_size[i]
                # covar_columns = delazify(covar_columns).squeeze(-2)
                covar_columns = [delazify(c).squeeze(-2) for c in covar_columns]
                if last_dim_is_batch:
                    covars = [ToeplitzLazyTensor(c.squeeze(dim=-2)) for c in covar_columns]
                else:
                    covars = [ToeplitzLazyTensor(c) for c in
                              covar_columns]  # TODO at least one of these two is not right, likely batch.
            else:
                full_covar = [self.base_kernel(grid[i], grid[i], **params) for i in range(n_dim)]
                if last_dim_is_batch:
                    covars = [full_covar]
                else:
                    covars = full_covar  # TODO: same message as above.

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
