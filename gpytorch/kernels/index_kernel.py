from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import torch
from gpytorch.kernels import Kernel

logger = logging.getLogger()


class IndexKernel(Kernel):
    def __init__(
        self,
        n_tasks,
        rank=1,
        covar_factor_prior=None,
        log_var_prior=None,
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
        logger.warning("Priors not yet implemented for IndexKernel. Ignoring bounds")
        self.register_parameter(
            name="covar_factor",
            parameter=torch.nn.Parameter(torch.randn(n_tasks, rank)),
            # prior=covar_factor_prior,
        )
        self.register_parameter(
            name="log_var",
            parameter=torch.nn.Parameter(torch.randn(n_tasks)),
            # prior=log_var_prior,
        )

    def forward(self, i1, i2):
        covar_matrix = self.covar_factor.matmul(self.covar_factor.transpose(-1, -2))
        covar_matrix += self.log_var.exp().diag()
        covar_matrix = covar_matrix.unsqueeze(0)
        output_covar = covar_matrix.index_select(-2, i1.view(-1)).index_select(
            -1, i2.view(-1)
        )
        return output_covar
