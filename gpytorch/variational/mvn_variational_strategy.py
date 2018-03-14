import torch
from .variational_strategy import VariationalStrategy
from ..lazy import LazyVariable, NonLazyVariable, RootLazyVariable


class MVNVariationalStrategy(VariationalStrategy):
    def kl_divergence(self):
        prior_mean = self.prior_dist.mean()
        prior_covar = self.prior_dist.covar()
        if not isinstance(prior_covar, LazyVariable):
            prior_covar = NonLazyVariable(prior_covar)
        prior_covar = prior_covar.add_jitter()

        variational_mean = self.variational_dist.mean()
        variational_covar = self.variational_dist.covar()
        if not isinstance(variational_covar, RootLazyVariable):
            raise RuntimeError('The variational covar for an MVN distribution should be a RootLazyVariable')
        chol_variational_covar = variational_covar.root.evaluate()

        mean_diffs = prior_mean - variational_mean
        inv_quad_rhs = torch.cat([chol_variational_covar.transpose(-1, -2), mean_diffs.unsqueeze(-1)], -1)

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
        trace_plus_inv_quad_form, logdet_prior_covar = prior_covar.inv_quad_log_det(inv_quad_rhs=inv_quad_rhs,
                                                                                    log_det=True)

        # Compute the KL Divergence.
        res = 0.5 * sum([
            logdet_prior_covar,
            logdet_variational_covar.mul(-1),
            trace_plus_inv_quad_form,
            -float(mean_diffs.size(-1)),
        ])
        return res
