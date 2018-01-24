import torch
import gpytorch
from .variational_strategy import VariationalStrategy
from ..lazy import RootLazyVariable


class MVNVariationalStrategy(VariationalStrategy):
    def kl_divergence(self):
        prior_mean = self.prior_dist.mean()
        prior_covar = self.prior_dist.covar()
        variational_mean = self.variational_dist.mean()
        variational_covar = self.variational_dist.covar()
        if not isinstance(variational_covar, RootLazyVariable):
            raise RuntimeError('The variational covar for an MVN distribution should be a RootLazyVariable')
        chol_variational_covar = variational_covar.root.evaluate()

        mean_diffs = prior_mean - variational_mean
        chol_variational_covar = chol_variational_covar

        if chol_variational_covar.ndimension() == 2:
            matrix_diag = chol_variational_covar.diag()
        elif chol_variational_covar.ndimension() == 3:
            batch_size, diag_size, _ = chol_variational_covar.size()
            batch_index = chol_variational_covar.data.new(batch_size).long()
            torch.arange(0, batch_size, out=batch_index)
            batch_index = batch_index.unsqueeze(1).repeat(1, diag_size).view(-1)
            diag_index = chol_variational_covar.data.new(diag_size).long()
            torch.arange(0, diag_size, out=diag_index)
            diag_index = diag_index.unsqueeze(1).repeat(batch_size, 1).view(-1)
            matrix_diag = chol_variational_covar[batch_index, diag_index, diag_index].view(batch_size, diag_size)
        else:
            raise RuntimeError('Invalid number of variational covar dimensions')

        logdet_variational_covar = matrix_diag.log().sum() * 2
        trace_logdet_quad_form = gpytorch.trace_logdet_quad_form(mean_diffs, chol_variational_covar,
                                                                 gpytorch.add_jitter(prior_covar))

        # Compute the KL Divergence.
        res = 0.5 * (trace_logdet_quad_form - logdet_variational_covar - len(mean_diffs))
        return res
