#!/usr/bin/env python3

import torch
from ..lazy import DiagLazyTensor, MatmulLazyTensor, SumLazyTensor
from ..distributions import MultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class VariationalStrategy(_VariationalStrategy):
    """
    """
    @cached(name="cholesky_factor")
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(induc_induc_covar.evaluate().double())
        return L

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        zeros = torch.zeros_like(self.inducing_points[..., 0])
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

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
        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            x = x.expand(*batch_shape, *x.shape[-2:])

        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        induc_mean = full_output.mean[..., :num_induc]
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(1e-4)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        # Standard version: for arbitrary means
        if induc_mean.sum().item() > 1e-10:
            solve_rhss = torch.cat([induc_data_covar, induc_mean.unsqueeze(-1)], -1).double()
            solves = torch.triangular_solve(solve_rhss, L, upper=False)[0].to(full_inputs.dtype)
            interp_term = solves[..., :-1]
            scaled_induc_mean = solves[..., -1]
        # Fast version: for zero means
        else:
            interp_term = torch.triangular_solve(induc_data_covar.double(), L, upper=False)[0].to(full_inputs.dtype)
            scaled_induc_mean = induc_mean  # Because both terms are equal to zero

        # Get q(u)
        variational_dist_u = self.variational_distribution

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = torch.matmul(
            interp_term.transpose(-1, -2),
            (variational_dist_u.mean - scaled_induc_mean).unsqueeze(-1)
        ).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        eye = torch.eye(num_induc, dtype=full_inputs.dtype, device=full_inputs.device)
        middle_term = (variational_dist_u.lazy_covariance_matrix.evaluate() - eye)
        predictive_covar = SumLazyTensor(
            data_data_covar.add_jitter(1e-4),
            MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term)
        )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
