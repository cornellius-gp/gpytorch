from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .grid_interpolation_kernel import GridInterpolationKernel


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

    for some kernel :math:`k'` that operates on each dimension.

    .. note::

        `AdditiveGridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    .. note::

        The :class:`~gpytorch.kernels.RBFKernel` decomposes multiplicatively. You should use
        `MultiplicativeGridInterpolationKernel` for multi-dimension RBF kernels!

    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`grid_size` (int):
            The size of the grid (in each dimension)
        :attr:`num_dims` (int):
            The dimension of the input data. Required if `grid_bounds=None`
        :attr:`grid_bounds` (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the size of the dim group (`num_dims // batch_dims`).
            The entries represent the min/max values for each dimension.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. Product Kernel Interpolation for Scalable Gaussian Processes:
        https://arxiv.org/pdf/1802.08903
    """

    def __init__(
        self,
        base_kernel,
        grid_size,
        num_dims=None,
        grid_bounds=None,
        active_dims=None,
    ):
        super(MultiplicativeGridInterpolationKernel, self).__init__(
            base_kernel=base_kernel,
            grid_size=grid_size,
            num_dims=num_dims,
            grid_bounds=grid_bounds,
            active_dims=active_dims,
        )

    def forward(self, x1, x2, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError(
                "MultiplicativeGridInterpolationKernel does not accept the batch_dims argument."
            )

        res = super(MultiplicativeGridInterpolationKernel, self).forward(x1, x2, batch_dims=(0, 2), **params)
        res = res.mul_batch(mul_batch_size=x1.size(-1))

        return res

    def __call__(self, x1_, x2_=None, diag=False, batch_dims=None, **params):
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
        return super(MultiplicativeGridInterpolationKernel, self).__call__(
            x1_, x2_, diag=diag, batch_dims=batch_dims, **params
        ).evaluate_kernel()
