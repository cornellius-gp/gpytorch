from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .kernel import Kernel
from ..lazy import DiagLazyVariable, InterpolatedLazyVariable, PsdSumLazyVariable, RootLazyVariable


def _eval_covar_matrix(covar_factor, log_var):
    return covar_factor.matmul(covar_factor.transpose(-1, -2)) + log_var.exp().diag()


class IndexKernel(Kernel):
    def __init__(self, n_tasks, rank=1, prior=None, active_dims=None):
        if rank > n_tasks:
            raise RuntimeError("Cannot create a task covariance matrix larger than the number of tasks")
        if active_dims is not None and len(active_dims) > 1:
            raise ValueError("Index must be with respect to a single column. Received {}".format(active_dims))
        super(IndexKernel, self).__init__(active_dims=active_dims)
        self.register_parameter(name="covar_factor", parameter=torch.nn.Parameter(torch.randn(n_tasks, rank)))
        self.register_parameter(name="log_var", parameter=torch.nn.Parameter(torch.randn(n_tasks)))
        if prior is not None:
            self.register_derived_prior(
                name="IndexKernelPrior",
                prior=prior,
                parameter_names=("covar_factor", "log_var"),
                transform=_eval_covar_matrix,
            )

    @property
    def covar_matrix(self):
        return PsdSumLazyVariable(RootLazyVariable(self.covar_factor), DiagLazyVariable(self.log_var.exp()))

    def forward(self, i1, i2):
        covar_matrix = _eval_covar_matrix(self.covar_factor, self.log_var)
        if covar_matrix.ndimension() == 2:
            covar_matrix = covar_matrix.unsqueeze(0)
        res = InterpolatedLazyVariable(base_lazy_variable=covar_matrix, left_interp_indices=i1, right_interp_indices=i2)
        return res
