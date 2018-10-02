from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .grid_interpolation_kernel import GridInterpolationKernel


class AdditiveGridInterpolationKernel(GridInterpolationKernel):
    r"""
    A variant of :class:`~gpytorch.kernels.GridInterpolationKernel` designed specifically
    for additive kernels. If a kernel decomposes additively, then this module will be much more
    computationally efficient.

    A kernel function `k` decomposes additively if it can be written as

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) + \ldots + k'(x_1^{(d)}, x_2^{(d)})
       \end{equation*}

    for some kernel :math:`k'` that operates on a subset of dimensions.

    The groupings of dimensions are specified by the :attr:`batch_dims` attribute.

    * `batch_dims=d` (d is the dimensionality of :math:`\mathbf x`):
        the kernel :math:`k` will be the sum of `d` sub-kernels, each operating
        on one dimension of :math:`\mathbf x`.

    * `batch_dims=d/2`:
        the first sub-kernel operates on dimensions 1 and 2, the second sub-kernel
        operates on dimensions 3 and 4, etc.

    * `batch_dims=1`:
        there is no additive decomposition

    .. note::

        `AdditiveGridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`grid_size` (int):
            The size of the grid (in each dimension)
        :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        :attr:`batch_dims` (int):
            The number of additive components
        :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the size of the dim group (`num_dims // batch_dims`).
            The entries represent the min/max values for each dimension.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
    """

    def __init__(
        self,
        base_kernel,
        grid_size,
        num_dims=None,
        grid_bounds=None,
        active_dims=None,
    ):
        super(AdditiveGridInterpolationKernel, self).__init__(
            base_kernel, grid_size, num_dims, grid_bounds, active_dims=active_dims
        )

    def forward(self, x1, x2, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError(
                "AdditiveGridInterpolationKernel does not accept the batch_dims argument."
            )

        res = super(AdditiveGridInterpolationKernel, self).forward(x1, x2, batch_dims=(0, 2), **params)
        res = res.sum_batch(sum_batch_size=x1.size(-1))
        return res
