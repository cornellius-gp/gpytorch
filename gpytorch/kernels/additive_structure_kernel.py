#!/usr/bin/env python3

from .kernel import Kernel


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

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if the base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(self, base_kernel, num_dims, active_dims=None):
        super(AdditiveStructureKernel, self).__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.num_dims = num_dims

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("AdditiveStructureKernel does not accept the last_dim_is_batch argument.")

        res = self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=True, **params)
        res = res.sum(-2 if diag else -3)
        return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return self.base_kernel.prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood)

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)
