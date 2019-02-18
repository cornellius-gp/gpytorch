#!/usr/bin/env python3

import math
import torch
from .. import settings, beta_features
from .variational_strategy import VariationalStrategy
from ..utils.memoize import cached
from ..lazy import RootLazyTensor, MatmulLazyTensor, CachedCGLazyTensor, DiagLazyTensor, BatchRepeatLazyTensor
from ..distributions import MultivariateNormal


class FullyTransformedVariationalStrategy(VariationalStrategy):
    def kl_divergence(self):
        variational_covar = self.variational_distribution.variational_distribution.covariance_matrix
        variational_mean = self.variational_distribution.variational_distribution.mean
        prior_dist = self.prior_distribution

        # Compute log|K| - log|S|
        prior_inv_variational = (prior_dist.lazy_covariance_matrix @ variational_covar)
        power = prior_inv_variational
        prior_inv_variational_log = prior_inv_variational

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, -0.5, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, 1. / 3, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, -1. / 4, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, 1. / 5, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, -1. / 6, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, 1. / 7, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, -1. / 8, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, 1. / 9, power)

        power = power @ prior_inv_variational
        prior_inv_variational_log = torch.add(prior_inv_variational_log, -1. / 10, power)

        logdets = prior_inv_variational_log.diagonal(dim1=-1, dim2=-2).sum(-1)

        # Compute covar trace
        covar_trace = prior_inv_variational.diagonal(dim1=-1, dim2=-2).sum(-1)

        # Compute mean_diff inv quad
        mean_diff_inv_quad = torch.sum((prior_dist.lazy_covariance_matrix @ variational_mean) * variational_mean, -1)

        print(logdets.item(), covar_trace.item(), mean_diff_inv_quad.item())

        kl_divergence = 0.5 * sum(
            [
                -logdets,
                covar_trace,
                mean_diff_inv_quad,
                # d
            ]
        )

        return kl_divergence

    def initialize_variational_dist(self):
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self.variational_distribution.chol_variational_covar.data.normal_().mul_(1e-8)
            self.variational_params_initialized.fill_(1)

    def forward(self, x):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        Args:
            x (torch.tensor): Locations x to get the variational posterior of the function values at.
        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(f|x)
        """
        variational_dist = self.variational_distribution.variational_distribution
        inducing_points = self.inducing_points
        if inducing_points.dim() < x.dim():
            inducing_points = inducing_points.expand(*x.shape[:-2], *inducing_points.shape[-2:])
            variational_dist = variational_dist.expand(x.shape[:-2])

        num_induc = inducing_points.size(-2)
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_mean, full_covar = full_output.mean, full_output.lazy_covariance_matrix

        # Mean terms
        test_mean = full_mean[..., num_induc:]
        induc_mean = full_mean[..., :num_induc]

        # Covariance terms
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].evaluate_kernel().add_jitter()
        data_induc_covar = full_covar[..., num_induc:, :num_induc]
        data_data_covar = full_covar[..., num_induc:, num_induc:]
        variational_covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate()

        # Compute predictive distribution
        predictive_mean = data_induc_covar @ variational_dist.mean + test_mean
        predictive_var = torch.sub(
            data_data_covar.diag(), 
            (data_induc_covar @ variational_covar_root).pow(2).sum(-1)
        )
        print(predictive_var.min().item())
        print(predictive_var.max().item())
        predictive_covar = DiagLazyTensor(predictive_var.clamp(1e-5, math.inf))

        # Save the logdet, mean_diff_inv_quad, prior distribution for the ELBO
        if self.training:
            self._memoize_cache["prior_distribution_memo"] = MultivariateNormal(induc_mean, induc_induc_covar)

        return MultivariateNormal(predictive_mean, predictive_covar)
