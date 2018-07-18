from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import torch
from gpytorch.kernels import Kernel

logger = logging.getLogger()


def _eval_covar_matrix(covar_factor, log_var):
    return covar_factor.matmul(covar_factor.transpose(-1, -2)) + log_var.exp().diag()


class IndexKernel(Kernel):
    def __init__(
        self,
        n_tasks,
        rank=1,
        prior=None,
        active_dims=None,
        covar_factor_bounds=(-100, 100),
        log_var_bounds=(-100, 100),
    ):
        if active_dims is not None and len(active_dims) > 1:
            raise ValueError(
                "Index must be with respect to a single column. Received {}".format(
                    active_dims
                )
            )
        super(IndexKernel, self).__init__(active_dims=active_dims)
        self.register_parameter(
            name="covar_factor",
            parameter=torch.nn.Parameter(torch.randn(n_tasks, rank)),
        )
        self.register_parameter(
            name="log_var",
            parameter=torch.nn.Parameter(torch.randn(n_tasks)),
        )
        if prior is not None:
            self.register_derived_prior(
                name="IndexKernelPrior",
                prior=prior,
                parameter_names=("covar_factor", "log_var"),
                transform=_eval_covar_matrix,
            )
        else:
            logger.warning("Cannot infer appropriate prior from bounds. Ignoring bounds.")

    def forward(self, i1, i2):
        covar_matrix = _eval_covar_matrix(self.covar_factor, self.log_var).unsqueeze(0)
        return covar_matrix.index_select(-2, i1.view(-1)).index_select(-1, i2.view(-1))
