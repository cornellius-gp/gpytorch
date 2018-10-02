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
        :attr:`inducing_points` (Tensor, n x d):
            This will be the set of points that lie on the grid.
        :attr:`grid` (Tensor, k x d):
            The exact grid points.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    def __init__(self, base_kernel, inducing_points, grid, active_dims=None):
        super(GridKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        if inducing_points.ndimension() != 2:
            raise RuntimeError("Inducing points should be 2 dimensional")
        self.register_buffer("inducing_points", inducing_points.unsqueeze(0))
        self.register_buffer("grid", grid)

    def train(self, mode=True):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return super(GridKernel, self).train(mode)

    def update_inducing_points_and_grid(self, inducing_points, grid):
        """
        Supply a new set of `inducing_points` and a new `grid` if they ever change.
        """
        self.inducing_points.detach().resize_(inducing_points.size()).copy_(inducing_points)
        self.grid.detach().resize_(grid.size()).copy_(grid)
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat
        return self

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        if not torch.equal(x1, self.inducing_points) or not torch.equal(x2, self.inducing_points):
            raise RuntimeError("The kernel should only receive the inducing points as input")

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
