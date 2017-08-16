import torch
from torch import nn
from .kernel import Kernel


class IndexKernel(Kernel):
    def __init__(self, n_tasks, rank=1, covar_factor_bounds=(-100, 100), log_var_bounds=(-100, 100)):
        super(IndexKernel, self).__init__()
        self.register_parameter('covar_factor', nn.Parameter(torch.zeros(n_tasks, rank)),
                                bounds=covar_factor_bounds)
        self.register_parameter('log_var', nn.Parameter(torch.zeros(n_tasks)), bounds=log_var_bounds)

    def forward(self, i1, i2):
        covar_matrix = self.covar_factor.mm(self.covar_factor.t()) + self.log_var.exp().diag()
        output_covar = covar_matrix.index_select(0, i1.view(-1)).index_select(1, i2.view(-1))
        return output_covar
