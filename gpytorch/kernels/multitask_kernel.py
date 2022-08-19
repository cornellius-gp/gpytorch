#!/usr/bin/env python3

from typing import Optional

from linear_operator import to_linear_operator
from linear_operator.operators import KroneckerProductLinearOperator

from ..priors import Prior
from .index_kernel import IndexKernel
from .kernel import Kernel


class MultitaskKernel(Kernel):
    r"""
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`~linear_operator.operators.KroneckerProductLinearOperator`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    :param ~gpytorch.priors.Prior task_covar_prior: (default None) Prior to use for task kernel.
        See :class:`gpytorch.kernels.IndexKernel` for details.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    def __init__(
        self,
        data_covar_module: Kernel,
        num_tasks: int,
        rank: Optional[int] = 1,
        task_covar_prior: Optional[Prior] = None,
        **kwargs,
    ):
        """"""
        super(MultitaskKernel, self).__init__(**kwargs)
        self.task_covar_module = IndexKernel(
            num_tasks=num_tasks, batch_shape=self.batch_shape, rank=rank, prior=task_covar_prior
        )
        self.data_covar_module = data_covar_module
        self.num_tasks = num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar_module.covar_matrix
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = to_linear_operator(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLinearOperator(covar_x, covar_i)
        return res.diagonal(dim1=-1, dim2=-2) if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks
