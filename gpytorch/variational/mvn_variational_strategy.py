from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from .variational_strategy import VariationalStrategy


class MVNVariationalStrategy(VariationalStrategy):
    def kl_divergence(self):
        return torch.distributions.kl.kl_divergence(self.variational_dist, self.prior_dist)

    def trace_diff(self):
        prior_covar = self.prior_dist.lazy_covariance_matrix
        variational_covar = self.variational_dist.lazy_covariance_matrix
        return (prior_covar.diag() - variational_covar.diag()).sum()
