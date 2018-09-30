from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
from .grid_interpolation_kernel import GridInterpolationKernel
from ..utils.interpolation import Interpolation


class MultiplicativeGridInterpolationKernel(GridInterpolationKernel):
    r"""
    A variant of :class:`~gpytorch.kernels.GridInterpolationKernel` designed specifically
    for kernels with product structure. If a kernel decomposes
    multiplicatively, then this module will be much more computationally efficient.
    See `Product Kernel Interpolation for Scalable Gaussian Processes`_ for more detail.

    A kernel function `k` has product structure if it can be written as

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) * \ldots * k'(x_1^{(d)}, x_2^{(d)})
       \end{equation*}

    for some kernel :math:`k'` that operates on a subset of dimensions.

    The groupings of dimensions are specified by the :attr:`dim_groups` attribute.

    * `dim_groups=d` (d is the dimensionality of :math:`\mathbf x`):
        the kernel :math:`k` will be the sum of `d` sub-kernels, each operating on one dimension of :math:`\mathbf x`.
    * `dim_groups=d/2`:
        the first sub-kernel operates on dimensions 1 and 2, the second sub-kernel
        operates on dimensions 3 and 4, etc.
    * `dim_groups=1`:
        there is no multiplicative decomposition

    .. note::

        `AdditiveGridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    .. note::

        The :class:`~gpytorch.kernels.RBFKernel` decomposes multiplicatively. You should use
        `MultiplicativeGridInterpolationKernel` for multi-dimension RBF kernels!

    Args:
        :attr:`base_kernel_module` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`grid_size` (int):
            The size of the grid (in each dimension)
        :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        :attr:`dim_groups` (int):
            The number of multiplicative components
        :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the size of the dim group (`num_dims // dim_groups`).
            The entries represent the min/max values for each dimension.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel_module`.

    .. Product Kernel Interpolation for Scalable Gaussian Processes:
        https://arxiv.org/pdf/1802.08903
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

        super(MultiplicativeGridInterpolationKernel, self).__init__(
            base_kernel_module=base_kernel_module,
            grid_size=grid_size,
            num_dims=(num_dims // dim_groups),
            grid_bounds=grid_bounds,
            active_dims=active_dims,
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
        res = super(MultiplicativeGridInterpolationKernel, self)._inducing_forward()
        return res.repeat(self.dim_groups, 1, 1)

    def forward(self, x1, x2):
        res = super(MultiplicativeGridInterpolationKernel, self).forward(x1, x2)
        res = res.mul_batch(mul_batch_size=self.dim_groups)
        return res

    def __call__(self, x1_, x2_=None, **params):
        """
        We cannot lazily evaluate actual kernel calls when using SKIP, because we
        cannot root decompose rectangular matrices.

        Because we slice in to the kernel during prediction to get the test x train
        covar before calling evaluate_kernel, the order of operations would mean we
        would get a MulLazyTensor representing a rectangular matrix, which we
        cannot matmul with because we cannot root decompose it. Thus, SKIP actually
        *requires* that we work with the full (train + test) x (train + test)
        kernel matrix.
        """
        return super(MultiplicativeGridInterpolationKernel, self).__call__(x1_, x2_, **params).evaluate_kernel()
