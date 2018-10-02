from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .variational_strategy import VariationalStrategy
from ..lazy import LazyTensor, NonLazyTensor


class MVNVariationalStrategy(VariationalStrategy):
    def kl_divergence(self):
        prior_mean = self.prior_dist.mean
        prior_covar = self.prior_dist.lazy_covariance_matrix
        if not isinstance(prior_covar, LazyTensor):
            prior_covar = NonLazyTensor(prior_covar)
        prior_covar = prior_covar.add_jitter()

        variational_mean = self.variational_dist.mean
        variational_covar = self.variational_dist.lazy_covariance_matrix
        root_variational_covar = variational_covar.root_decomposition()

        mean_diffs = prior_mean - variational_mean
        inv_quad_rhs = torch.cat([root_variational_covar, mean_diffs.unsqueeze(-1)], -1)
        log_det_variational_covar = variational_covar.log_det()
        trace_plus_inv_quad_form, log_det_prior_covar = prior_covar.inv_quad_log_det(
            inv_quad_rhs=inv_quad_rhs, log_det=True
        )

        # Compute the KL Divergence.
        res = 0.5 * sum(
            [
                log_det_prior_covar,
                log_det_variational_covar.mul(-1),
                trace_plus_inv_quad_form,
                -float(mean_diffs.size(-1)),
            ]
        )
        return res

    def trace_diff(self):
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        return (prior_covar.diag() - variational_covar.diag()).sum()
