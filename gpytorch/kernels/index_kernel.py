from .kernel import Kernel


class IndexKernel(Kernel):
    def forward(self, i1, i2, index_covar_factor, index_log_var):
        index_covar_matrix = index_covar_factor.mm(index_covar_factor.t()) + index_log_var.exp().diag()
        output_covar = index_covar_matrix.index_select(0, i1.view(-1)).index_select(1, i2.view(-1))
        return output_covar
