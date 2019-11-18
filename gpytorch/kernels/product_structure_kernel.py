#!/usr/bin/env python3

from ..lazy import lazify
from .kernel import Kernel


class ProductStructureKernel(Kernel):
    r"""
    A Kernel decorator for kernels with product structure. If a kernel decomposes
    multiplicatively, then this module will be much more computationally efficient.

    A kernel function `k` has product structure if it can be written as

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) * \ldots * k'(x_1^{(d)}, x_2^{(d)})
       \end{equation*}

    for some kernel :math:`k'` that operates on each dimension.

    Given a `b x n x d` input, `ProductStructureKernel` computes `d` one-dimensional kernels
    (using the supplied base_kernel), and then multiplies the component kernels together.
    Unlike :class:`~gpytorch.kernels.ProductKernel`, `ProductStructureKernel` computes each
    of the product terms in batch, making it very fast.

    See `Product Kernel Interpolation for Scalable Gaussian Processes`_ for more detail.

    Args:
        - :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        - :attr:`num_dims` (int):
            The dimension of the input data.
        - :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Product Kernel Interpolation for Scalable Gaussian Processes:
        https://arxiv.org/pdf/1802.08903
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if the base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, num_dims, active_dims=None):
        super(ProductStructureKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.num_dims = num_dims

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ProductStructureKernel does not accept the last_dim_is_batch argument.")

        res = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=True, **params)
        res = res.prod(-2 if diag else -3)
        return res

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def __call__(self, x1_, x2_=None, diag=False, last_dim_is_batch=False, **params):
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
        res = super().__call__(x1_, x2_, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
        res = lazify(res).evaluate_kernel()
        return res
