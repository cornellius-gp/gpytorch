#!/usr/bin/env python3

import warnings

import torch

from torch.distributions.kl import kl_divergence
from ..distributions import MultivariateNormal, Delta
from ..lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, delazify
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class DecoupledVariationalStrategy(_VariationalStrategy):
    r"""
    """

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True,
                 return_separate_mvns=False):
        # We're going to create two set of inducing points
        # One set for computing the mean, one set for computing the variance
        self.return_separate_mvns = return_separate_mvns
        inducing_points = torch.stack([inducing_points, inducing_points])  # Create two copies
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)

    @cached(name="cholesky_factor")
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double())
        return L

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros_like(self.variational_distribution.mean)
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        # We'll compute the covariance, and cross-covariance terms for both the
        # pred-mean and pred-covar, using their different inducing points (and maybe kernel hypers)

        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        interp_term = torch.triangular_solve(induc_data_covar.double(), L, upper=False)[0].to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} m + \mu_X
        # Here we're using the terms that correspond to the mean's inducing points
        predictive_mean = (
            torch.matmul(
                interp_term[0].transpose(-1, -2), inducing_values.unsqueeze(-1)
            ).squeeze(-1) + test_mean[0]
        )

        if self.model.training and self.return_separate_mvns: # VARIATIONAL FITC
            predictive_covar1 = SumLazyTensor(data_data_covar.add_jitter(1e-4).evaluate()[1],
                                              MatmulLazyTensor(interp_term[1].transpose(-1, -2),
                                                               self.prior_distribution.lazy_covariance_matrix.mul(-1) \
                                                               @ interp_term[1]))
            predictive_covar2 = MatmulLazyTensor(interp_term[1].transpose(-1, -2), variational_inducing_covar @ interp_term[1])

            return MultivariateNormal(predictive_mean, predictive_covar1), \
                   MultivariateNormal(predictive_mean, predictive_covar2)
        else:  # not FITC
            # Compute the covariance of q(f)
            # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
            middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
            if variational_inducing_covar is not None:
                middle_term = SumLazyTensor(variational_inducing_covar, middle_term)
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4).evaluate()[1],
                MatmulLazyTensor(interp_term[1].transpose(-1, -2), middle_term @ interp_term[1])
            )

            return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        variational_dist = self.variational_distribution
        prior_dist = self.variational_distribution

        mean_dist = Delta(variational_dist.mean)
        covar_dist = MultivariateNormal(
            torch.zeros_like(variational_dist.mean),
            variational_dist.lazy_covariance_matrix
        )
        return kl_divergence(mean_dist, prior_dist) + kl_divergence(covar_dist, prior_dist)
