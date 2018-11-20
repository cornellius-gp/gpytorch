#!/usr/bin/env python3

from .kernel import Kernel
from ..lazy import LazyTensor, NonLazyTensor


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

    def __init__(self, base_kernel, num_dims, active_dims=None):
        super(ProductStructureKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.num_dims = num_dims

    def forward(self, x1, x2, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError("ProductStructureKernel does not accept the batch_dims argument.")

        res = self.base_kernel(x1, x2, batch_dims=(0, 2), **params).evaluate_kernel()

        evaluate = False
        if not isinstance(res, LazyTensor):
            evaluate = True
            res = NonLazyTensor(res)

        res = res.mul_batch(mul_batch_size=x1.size(-1))

        if evaluate:
            res = res.evaluate()
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
        return (
            super(ProductStructureKernel, self)
            .__call__(x1_, x2_, diag=diag, batch_dims=batch_dims, **params)
            .evaluate_kernel()
        )
