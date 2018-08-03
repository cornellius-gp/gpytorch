from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from .index_kernel import IndexKernel
from ..lazy import LazyVariable, NonLazyVariable, KroneckerProductLazyVariable


def _eval_covar_matrix(covar_factor, log_var):
    return covar_factor.matmul(covar_factor.transpose(-1, -2)) + log_var.exp().diag()


class MultitaskKernel(Kernel):
    def __init__(self, data_covar_module, n_tasks, rank=1, task_covar_prior=None):
        super(MultitaskKernel, self).__init__()
        self.task_covar_module = IndexKernel(n_tasks=n_tasks, rank=rank, prior=task_covar_prior)
        self.data_covar_module = data_covar_module
        self.n_tasks = n_tasks

    def forward_diag(self, x1, x2):
        x1 = x1.transpose(-2, -3)
        x2 = x2.transpose(-2, -3)
        task_indices = torch.range(0, self.n_tasks - 1, device=x1.device).long()
        task_indices = task_indices.unsqueeze(-1).unsqueeze(-1)
        covar_i = self.task_covar_module(task_indices).evaluate_kernel()
        covar_x = self.data_covar_module.forward(x1, x2).squeeze(0)
        if not isinstance(covar_x, LazyVariable):
            covar_x = NonLazyVariable(covar_x)
        res = KroneckerProductLazyVariable(covar_i, covar_x).matmul(torch.eye(1, device=x1.device)).unsqueeze(-1)
        return res

    def forward(self, x1, x2):
        task_indices = torch.range(0, self.n_tasks - 1, device=x1.device).long()
        covar_i = self.task_covar_module(task_indices).evaluate_kernel()
        covar_x = self.data_covar_module.forward(x1, x2).squeeze(0)
        if not isinstance(covar_x, LazyVariable):
            covar_x = NonLazyVariable(covar_x)
        res = KroneckerProductLazyVariable(covar_i, covar_x)
        return res

    def size(self, x1, x2):
        if x1.ndimension() == 3:
            return torch.Size((x1.size(0), self.n_tasks * x1.size(-2), self.n_tasks * x2.size(-2)))
        else:
            return torch.Size((self.n_tasks * x1.size(-2), self.n_tasks * x2.size(-2)))
