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
        :attr:`grid` (Tensor, k x d):
            The exact grid points.
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

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        """
        Forward the grid data through the base kernel.

        **Note:** GridKernel entirely ignores x1 and x2, and only forwards the grid through the base kernel.
        If for some reason you need to change the grid, use the update_grid method.
        """
        if not self.training and hasattr(self, "_cached_kernel_mat"):
            covar = self._cached_kernel_mat

        else:
            n_dim = x1.size(-1)
            grid = self.grid.unsqueeze(0)

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
