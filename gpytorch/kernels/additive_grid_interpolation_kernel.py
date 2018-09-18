from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
from .grid_interpolation_kernel import GridInterpolationKernel
from ..utils import Interpolation


class AdditiveGridInterpolationKernel(GridInterpolationKernel):
    r"""
    A variant of :class:`~gpytorch.kernels.GridInterpolationKernel` designed specifically
    for additive kernels. If a kernel decomposes additively, then this module will be much more
    computationally efficient.

    A kernel function `k` decomposes additively if it can be written as

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) + \ldots + k'(x_1^({d}), x_2^{(d)})
       \end{equation*}

    for some kernel :math:`k'` that operates on a subset of dimensions.

    The groupings of dimensions are specified by the :attr:`dim_groups` attribute.
    * `dim_groups=d` (d is the dimensionality of :math:`\mathbf x`): the kernel
        :math:`k` will be the sum of `d` sub-kernels, each operating on one dimension of :math:`\mathbf x`.
    * `dim_groups=d/2`: the first sub-kernel operates on dimensions 1 and 2, the second sub-kernel
        operates on dimensions 3 and 4, etc.
    * `dim_groups=1`: there is no additive decomposition

    .. note::

        `AdditiveGridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        :attr:`base_kernel_module` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`grid_size` (int):
            The size of the grid (in each dimension)
        :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        :attr:`dim_groups` (int):
            The number of additive components
        :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the size of the dim group (num_dims // dim_groups).
            The entries represent the min/max values for each dimension.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel_module`.
    """

    def __init__(
        self,
        base_kernel_module,
        grid_size,
        dim_groups=None,
        num_dims=None,
        grid_bounds=None,
        active_dims=None,
        n_components=None,
    ):
        if n_components is not None:
            warnings.warn("n_components is deprecated. Use dim_groups instead.", DeprecationWarning)
            dim_groups = n_components
        if dim_groups is None:
            raise RuntimeError("Must supply dim_groups")

        super(AdditiveGridInterpolationKernel, self).__init__(
            base_kernel_module, grid_size, num_dims // dim_groups, grid_bounds, active_dims=active_dims
        )

        self.dim_groups = dim_groups

    def _compute_grid(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1), self.dim_groups, -1)
        batch_size, n_data, dim_groups, n_dimensions = inputs.size()
        inputs = inputs.transpose(0, 2).contiguous().view(dim_groups * batch_size * n_data, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        interp_indices = interp_indices.view(dim_groups * batch_size, n_data, -1)
        interp_values = interp_values.view(dim_groups * batch_size, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self):
        res = super(AdditiveGridInterpolationKernel, self)._inducing_forward()
        return res.repeat(self.dim_groups, 1, 1)

    def forward(self, x1, x2):
        res = super(AdditiveGridInterpolationKernel, self).forward(x1, x2)
        return res.sum_batch(sum_batch_size=self.dim_groups)
