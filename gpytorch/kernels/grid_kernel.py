from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from ..lazy import ToeplitzLazyTensor, KroneckerProductLazyTensor
from .. import settings


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

    def __init__(self, base_kernel, grid, active_dims=None):
        super(GridKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.register_buffer("grid", grid)

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        self.grid.detach().resize_(grid.size()).copy_(grid)
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    def _create_data_from_grid(self):
        grid_size = self.grid.size(-2)
        grid_dim = self.grid.size(-1)
        grid_data = torch.zeros(int(pow(grid_size, grid_dim)), grid_dim)
        prev_points = None
        for i in range(grid_dim):
            for j in range(grid_size):
                grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(self.grid[j, i])
                if prev_points is not None:
                    grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
            prev_points = grid_data[: grid_size ** (i + 1), : (i + 1)]

        return grid_data

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        grid = self.grid.unsqueeze(0)
        x1 = x1.unsqueeze(0) if x1.dim() == 2 else x1
        x2 = x2.unsqueeze(0) if x2.dim() == 2 else x2
        if torch.equal(x1, grid) and torch.equal(x2, grid):

            if not self.training and hasattr(self, "_cached_kernel_mat"):
                return self._cached_kernel_mat

            n_dim = x1.size(-1)

            if settings.use_toeplitz.on():
                first_item = grid[:, 0:1]
                covar_columns = self.base_kernel(first_item, grid, diag=False, batch_dims=(0, 2), **params)
                covar_columns = covar_columns.evaluate().squeeze(-2)
                if batch_dims == (0, 2):
                    covars = [ToeplitzLazyTensor(covar_columns)]
                else:
                    covars = [ToeplitzLazyTensor(covar_columns[i : i + 1]) for i in range(n_dim)]
            else:
                full_covar = self.base_kernel(grid, grid, batch_dims=(0, 2), **params).evaluate_kernel()
                if batch_dims == (0, 2):
                    covars = [full_covar]
                else:
                    covars = [full_covar[i : i + 1] for i in range(n_dim)]

            if len(covars) > 1:
                covar = KroneckerProductLazyTensor(*covars)
            else:
                covar = covars[0]

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
        else:
            x1_ = self._create_data_from_grid() if torch.equal(x1, grid) else x1
            x2_ = self._create_data_from_grid() if torch.equal(x2, grid) else x2
            return self.base_kernel.forward(x1_, x2_, diag=diag, batch_dims=batch_dims, **params)

    def size(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariancn matrix.
        """
        if x1.dim() == 3:
            grid = self.grid.unsqueeze(0)
        else:
            grid = self.grid
        left_size = pow(grid.size(-2), grid.size(-1)) if torch.equal(x1, grid) else x1.size(-2)
        right_size = pow(grid.size(-2), grid.size(-1)) if torch.equal(x2, grid) else x2.size(-2)
        non_batch_size = (left_size, right_size)
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)
