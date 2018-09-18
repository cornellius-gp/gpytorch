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
    Implements the KISS-GP (or SKI) approximation for a given kernel.
    It was proposed in `Kernel Interpolation for Scalable Structured Gaussian Processes`_,
    and offers extremely fast and accurate Kernel approximations for large datasets.

    Given a base kernel `k`, the covariance :math:`k(\mathbf{x_1}, \mathbf{x_2})` is approximated by
    using a grid of regularly spaced *inducing points*:

    .. note::

        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        :attr:`base_kernel_module` (Kernel):
            The kernel to speed up with grid methods.
        :attr:`inducing_points` (Tensor, n x d):
            This will be the set of points that lie on the grid.
        :attr:`grid` (Tensor, k x d):
            The exact grid points.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel_module`.

    .. Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    def __init__(self, base_kernel_module, inducing_points, grid, active_dims=None):
        super(GridKernel, self).__init__(active_dims=active_dims)
        self.base_kernel_module = base_kernel_module
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

    def forward(self, x1, x2, **kwargs):
        if not torch.equal(x1, self.inducing_points) or not torch.equal(x2, self.inducing_points):
            raise RuntimeError("The kernel should only receive the inducing points as input")

        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat

        else:
            n_dim = x1.size(-1)
            grid_var = self.grid.view(n_dim, -1, 1)

            if settings.use_toeplitz.on():
                first_item = grid_var[:, 0:1].contiguous()
                covar_columns = self.base_kernel_module(first_item, grid_var, **kwargs).evaluate()
                covars = [ToeplitzLazyTensor(covar_columns[i : i + 1].squeeze(-2)) for i in range(n_dim)]
            else:
                grid_var = grid_var.view(n_dim, -1, 1)
                covars = self.base_kernel_module(grid_var, grid_var, **kwargs).evaluate_kernel()
                covars = [covars[i : i + 1] for i in range(n_dim)]

            if n_dim > 1:
                covar = KroneckerProductLazyTensor(*covars)
            else:
                covar = covars[0]

            if not self.training:
                self._cached_kernel_mat = covar

            return covar
