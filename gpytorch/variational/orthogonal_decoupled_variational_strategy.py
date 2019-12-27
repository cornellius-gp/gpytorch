#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class OrthogonalDecoupledVariationalStrategy(_VariationalStrategy):
    r"""
    As defined in `Salimbeni et al. (2018)`_.

    .. _Salimbeni et al. (2018):
        https://arxiv.org/abs/1809.08820
    """

    def __init__(self, model, inducing_points, variational_distribution):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations=True)
        self.base_variational_strategy = model

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        if variational_inducing_covar is not None:
            raise NotImplementedError("DecoupledVariationalStrategy currently works with DeltaVariationalDistribution")

        num_data = x.size(-2)
        full_output = self.model(torch.cat([x, inducing_points], dim=-2))
        full_mean = full_output.mean
        full_covar = full_output.lazy_covariance_matrix

        # Cache the kernel matrix with the cached CG calls
        if self.training:
            induc_mean = full_mean[..., num_data:]
            induc_induc_covar = full_covar[..., num_data:, num_data:]
            self._memoize_cache["prior_distribution_memo"] = MultivariateNormal(induc_mean, induc_induc_covar)

        test_mean = full_mean[..., :num_data]
        data_induc_covar = full_covar[..., :num_data, num_data:]
        predictive_mean = (data_induc_covar @ inducing_values.unsqueeze(-1)).squeeze(-1).add(test_mean)
        predictive_covar = full_covar[..., :num_data, :num_data]

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        mean = self.variational_distribution.mean
        induc_induc_covar = self.prior_distribution.lazy_covariance_matrix
        kl = self.model.kl_divergence() + ((induc_induc_covar @ mean.unsqueeze(-1)).squeeze(-1) * mean).sum(-1)
        return kl
