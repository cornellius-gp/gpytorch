from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from .index_kernel import IndexKernel
from ..lazy import LazyVariable, NonLazyVariable, KroneckerProductLazyVariable


class MultitaskKernel(Kernel):
    """
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyVariable`.
    """

    def __init__(self, data_covar_module, n_tasks, rank=1, task_covar_prior=None):
        """
        Args:
            data_covar_module (:obj:`gpytorch.kernels.Kernel`): Kernel to use as the data kernel.
            n_tasks (int): Number of tasks
            rank (int): Rank of index kernel to use for task covariance matrix.
            task_covar_prior (:obj:`gpytorch.priors.Prior`): Prior to use for task kernel. See
                :class:`gpytorch.kernels.IndexKernel` for details.
        """
        super(MultitaskKernel, self).__init__()
        self.task_covar_module = IndexKernel(n_tasks=n_tasks, rank=rank, prior=task_covar_prior)
        self.data_covar_module = data_covar_module
        self.n_tasks = n_tasks

    def forward_diag(self, x1, x2):
        """
        Returns the diagonal of the covariance matrix only. This overrides the default behavior for this supplied
        in :class:`gpytorch.kernels.Kernel` because we need to take the Kronecker product of the diagonals of the
        base data kernel and the task kernel.
        """
        task_indices = torch.arange(self.n_tasks, device=x1.device).long()
        task_indices = task_indices.unsqueeze(0).unsqueeze(-1)

        # These are small because they are vectors, therefore it is safe to evaluate them
        covar_i_diag = self.task_covar_module.forward_diag(task_indices, task_indices)
        covar_x_diag = self.data_covar_module.forward_diag(x1, x2)

        if isinstance(covar_x_diag, LazyVariable):
            covar_x_diag = covar_x_diag.evaluate()
        if isinstance(covar_i_diag, LazyVariable):
            covar_i_diag = covar_i_diag.evaluate()

        covar_x_diag = covar_x_diag.squeeze(-1)
        covar_i_diag = covar_i_diag.squeeze(-1)

        # Take the Kronecker product of the two diagonals
        res = KroneckerProductLazyVariable(NonLazyVariable(covar_i_diag), NonLazyVariable(covar_x_diag)).evaluate()
        return res.unsqueeze(-1)

    def forward(self, x1, x2):
        covar_i = self.task_covar_module.covar_matrix
        covar_x = self.data_covar_module.forward(x1, x2)
        if covar_x.size(0) == 1:
            covar_x = covar_x[0]
        if not isinstance(covar_x, LazyVariable):
            covar_x = NonLazyVariable(covar_x)
        res = KroneckerProductLazyVariable(covar_i, covar_x)
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
