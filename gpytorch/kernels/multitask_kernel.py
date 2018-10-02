from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from .index_kernel import IndexKernel
from ..lazy import LazyTensor, NonLazyTensor, KroneckerProductLazyTensor


class MultitaskKernel(Kernel):
    """
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    Args:
        data_covar_module (:obj:`gpytorch.kernels.Kernel`):
            Kernel to use as the data kernel.
        n_tasks (int):
            Number of tasks
        batch_size (int, optional):
            Set if the MultitaskKernel is operating on batches of data (and you want different
            parameters for each batch)
        rank (int):
            Rank of index kernel to use for task covariance matrix.
        task_covar_prior (:obj:`gpytorch.priors.Prior`):
            Prior to use for task kernel. See :class:`gpytorch.kernels.IndexKernel` for details.
    """

    def __init__(self, data_covar_module, n_tasks, rank=1, batch_size=1, task_covar_prior=None):
        """
        """
        super(MultitaskKernel, self).__init__()
        self.task_covar_module = IndexKernel(n_tasks=n_tasks, batch_size=batch_size, rank=rank, prior=task_covar_prior)
        self.data_covar_module = data_covar_module
        self.n_tasks = n_tasks
        self.batch_size = 1

    def forward(self, x1, x2, diag=False, batch_dims=None, **params):
        if batch_dims == (0, 2):
            raise RuntimeError("AdditiveGridInterpolationKernel does not accept the batch_dims argument.")

        covar_i = self.task_covar_module.covar_matrix
        covar_i = covar_i.repeat(x1.size(0), 1, 1)
        covar_x = self.data_covar_module(x1, x2, **params)
        if not isinstance(covar_x, LazyTensor):
            covar_x = NonLazyTensor(covar_x)
        res = KroneckerProductLazyTensor(covar_i, covar_x)

        if diag:
            return res.diag()
        else:
            return res

    def size(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask kernel returns an `(n*n_tasks) x (m*n_tasks)`
        covariance matrix.
        """
        non_batch_size = (self.n_tasks * x1.size(-2), self.n_tasks * x2.size(-2))
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0),) + non_batch_size)
        else:
            return torch.Size(non_batch_size)
