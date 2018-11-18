#!/usr/bin/env python3

from .kernel import Kernel
from ..lazy import LazyTensor, NonLazyTensor


class AdditiveStructureKernel(Kernel):
    r"""
    A Kernel decorator for kernels with additive structure. If a kernel decomposes
    additively, then this module will be much more computationally efficient.

    A kernel function `k` decomposes additively if it can be written as

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = k'(x_1^{(1)}, x_2^{(1)}) + \ldots + k'(x_1^{(d)}, x_2^{(d)})
       \end{equation*}

    for some kernel :math:`k'` that operates on a subset of dimensions.

    Given a `b x n x d` input, `AdditiveStructureKernel` computes `d` one-dimensional kernels
    (using the supplied base_kernel), and then adds the component kernels together.
    Unlike :class:`~gpytorch.kernels.AdditiveKernel`, `AdditiveStructureKernel` computes each
    of the additive terms in batch, making it very fast.

    Args:
        :attr:`base_kernel` (Kernel):
            The kernel to approximate with KISS-GP
        :attr:`num_dims` (int):
            The dimension of the input data.
        :attr:`active_dims` (tuple of ints, optional):
            Passed down to the `base_kernel`.
    """

    def __init__(self, base_kernel, num_dims, active_dims=None):
        super(AdditiveStructureKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.num_dims = num_dims

    def forward(self, x1, x2, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError("AdditiveStructureKernel does not accept the batch_dims argument.")

        res = self.base_kernel(x1, x2, batch_dims=(0, 2), **params).evaluate_kernel()

        evaluate = False
        if not isinstance(res, LazyTensor):
            evaluate = True
            res = NonLazyTensor(res)

        res = res.sum_batch(sum_batch_size=x1.size(-1))

        if evaluate:
            res = res.evaluate()
        return res
