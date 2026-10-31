#!/usr/bin/env python3

from linear_operator import to_linear_operator
from linear_operator.operators import BlockInterleavedLinearOperator

from .kernel import Kernel


class ParallelPartialKernel(Kernel):
    r"""
    A special :class:`gpytorch.kernels.MultitaskKernel` where tasks are assumed
    to be independent, and a single, common kernel is used for all tasks.

    Given a base covariance module to be used for the data, :math:`K_{XX}`,
    this kernel returns :math:`K = I_T \otimes K_{XX}`, where :math:`T` is the
    number of tasks.

    .. note::

        Note that, in this construction, it is crucial that all coordinates (or
        tasks) share the same kernel, with the same kernel parameters. The
        simplification of the inter-task kernel leads to computational
        savings if the number of tasks is large. If this were not the case
        (for example, when using the batch-independent Gaussian Process
        construction), then each task would have a different design correlation
        matrix, requiring the inversion of an `n x n` matrix at each
        coordinate, where `n` is the number of data points. Furthermore, when
        training the Gaussian Process surrogate, there is only one set of
        kernel parameters to be estimated, instead of one for every coordinate.

    :param ~gpytorch.kernels.Kernel covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks.
    :param dict kwargs: Additional arguments to pass to the kernel.

    Example:
    """

    def __init__(
        self,
        covar_module: Kernel,
        num_tasks: int,
        **kwargs,
    ):
        super(ParallelPartialKernel, self).__init__(**kwargs)
        self.covar_module = covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("ParallelPartialKernel does not accept the last_dim_is_batch argument.")
        covar_x = to_linear_operator(self.covar_module.forward(x1, x2, **params))
        res = BlockInterleavedLinearOperator(covar_x.repeat(self.num_tasks, 1, 1))
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this parallel
        partial kernel returns an `(n*num_tasks) x (m*num_tasks)`
        block-diagonal covariance matrix with `num_tasks` blocks of shape
        `n x m` on the diagonal.
        """
        return self.num_tasks
