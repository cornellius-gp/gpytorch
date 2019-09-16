#!/usr/bin/env python3

import torch
from .. import settings
from ..lazy import DiagLazyTensor, MatmulLazyTensor
from ..module import Module
from ..distributions import MultivariateNormal
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.cholesky import psd_safe_cholesky
from ..utils.memoize import cached
from ..utils.lanczos import lanczos_tridiag


class HalfWhitenedVariationalStrategy(Module):
    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=False):
        super(HalfWhitenedVariationalStrategy, self).__init__()
        object.__setattr__(self, "model", model)

        inducing_points = inducing_points.clone()

        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        self.variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    @cached(name="covar_trace_memo")
    def covar_trace(self):
        sqrt_covar = self.variational_distribution.variational_distribution.lazy_covariance_matrix.cholesky()
        return (sqrt_covar * sqrt_covar).sum(dim=-1).sum(dim=-1)

    @cached(name="mean_diff_inv_quad_memo")
    def mean_diff_inv_quad(self):
        prior_mean = self.prior_distribution.mean
        if prior_mean.dim() == 1:
            return torch.dot(prior_mean, prior_mean)
        return prior_mean.transpose(-2, -1).matmul(prior_mean)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.
        """
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(
            out.mean, out.lazy_covariance_matrix.add_jitter()
        )
        return res

    @cached(name="mean_covar_cache_memo")
    def mean_covar_cache(self):
        """
        Computes K_{uu}^{-1/2}m and K_{uu}^{-1/2}(I - LL')K_{uu}^{-1/2} using contour integral quadrature.
        """
        prior_dist = self.prior_distribution
        variational_dist_u = self.variational_distribution.variational_distribution

        induc_induc_covar = prior_dist.lazy_covariance_matrix

        L = induc_induc_covar.evaluate().cholesky()

        device = induc_induc_covar.device
        dtype = induc_induc_covar.dtype
        mat_len = induc_induc_covar.matrix_shape[0]
        batch_shape = induc_induc_covar.batch_shape

        eye = DiagLazyTensor(torch.ones(*batch_shape, mat_len, dtype=dtype, device=device))

        inner_mat = (eye + variational_dist_u.lazy_covariance_matrix.mul(-1)).evaluate()

        right_rinv = torch.triangular_solve(inner_mat, L.transpose(-2, -1), upper=True)[0].transpose(-2, -1)

        var_mean = variational_dist_u.mean - prior_dist.mean

        right_hand_sides = torch.cat((var_mean.unsqueeze(-1), right_rinv), dim=-1)

        full_rinv, _ = torch.triangular_solve(right_hand_sides, L.transpose(-2, -1), upper=True)

        mean_cache = full_rinv[..., :, 0].contiguous()
        covar_cache = full_rinv[..., :, 1:].contiguous()

        return [mean_cache, covar_cache]

    def kl_divergence(self):
        """
        Computes KL divergence where the variationa distribution q(u) is given by N(K^{1/2}m, K^{1/2}LL'K^{1/2'}).
        """
        variational_dist_u = self.variational_distribution.variational_distribution
        prior_dist = self.prior_distribution
        kl_divergence = 0.5 * sum(   # -2 log | L | + tr(LL') + m'm - k
            [
                -variational_dist_u.lazy_covariance_matrix.logdet(),
                self.covar_trace(),
                self.mean_diff_inv_quad(),
                -prior_dist.event_shape.numel(),
            ]
        )

        return kl_divergence

    def initialize_variational_dist(self):
        """
        Describes what distribution to pass to the VariationalDistribution to initialize with. Most commonly, this
        should be the prior distribution for the inducing points, N(m_u, K_uu). However, if a subclass assumes
        a different parameterization of the variational distribution, it may need to modify what the prior is
        with respect to that reparameterization.

        TODO for half whitening
        """
        prior_dist = self.prior_distribution
        induc_induc_covar = prior_dist.covariance_matrix
        chol_induc = induc_induc_covar.cholesky()
        inv_chol_induc = chol_induc.double().inverse().to(induc_induc_covar.dtype)

        init_mu = inv_chol_induc.matmul(prior_dist.mean.unsqueeze(-1)).squeeze(-1)
        init_covar = inv_chol_induc.matmul(induc_induc_covar.matmul(inv_chol_induc.transpose(-2, -1)))
        eval_prior_dist = torch.distributions.MultivariateNormal(
            loc=init_mu + 1e-1 * torch.randn_like(init_mu),
            scale_tril=init_covar,
        )

        self.variational_distribution.initialize_variational_distribution(eval_prior_dist)

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
        inducing_batch_shape = inducing_points.shape[:-2]
        if inducing_batch_shape < x.shape[:-2]:
            batch_shape = _mul_broadcast_shape(inducing_points.shape[:-2], x.shape[:-2])
            inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
            x = x.expand(*batch_shape, *x.shape[-2:])
            variational_dist = variational_dist.expand(batch_shape)

        # If our points equal the inducing points, we're done
        if torch.equal(x, inducing_points):
            return variational_dist

        # Otherwise, we have to marginalize
        else:
            num_induc = inducing_points.size(-2)
            full_inputs = torch.cat([inducing_points, x], dim=-2)
            full_output = self.model.forward(full_inputs)
            full_covar = full_output.lazy_covariance_matrix
            test_mean = full_output.mean[..., num_induc:]

            # Covariance terms
            induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()
            data_data_covar = full_covar[..., num_induc:, num_induc:]

            mean_cache, covar_cache = self.mean_covar_cache()

            predictive_mean = test_mean + induc_data_covar.transpose(-2, -1).matmul(mean_cache)

            left_part = induc_data_covar.transpose(-2, -1).matmul(covar_cache)
            full_part = MatmulLazyTensor(left_part, induc_data_covar)
            predictive_covar = data_data_covar + full_part.mul(-1)

            if self.training:
                predictive_covar = DiagLazyTensor(predictive_covar.diag())

            return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x):
        if not self.variational_params_initialized.item():
            self.initialize_variational_dist()
            self.variational_params_initialized.fill_(1)
        if self.training:
            if hasattr(self, "_memoize_cache"):
                delattr(self, "_memoize_cache")
                self._memoize_cache = dict()

        return super(HalfWhitenedVariationalStrategy, self).__call__(x)
