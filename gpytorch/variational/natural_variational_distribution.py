#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..lazy import CholLazyTensor
from ..utils.cholesky import psd_safe_cholesky
from ._variational_distribution import _VariationalDistribution


class _ExpectationParams(torch.autograd.Function):
    """
    Given the canonical params, computes the expectation params

    expec_mean = m = -1/2 natural_mat^{-1} natural_vec
    expec_covar = m m^T + S = [ expec_mean expec_mean^T - 0.5 natural_mat^{-1} ]

    On the backward pass, we simply pass the gradients from the expected parameters into the canonical params
    This will allow us to perform NGD
    """

    @staticmethod
    def forward(ctx, natural_vec, natural_mat):
        cov_mat = natural_mat.inverse().mul_(-0.5)
        expec_mean = cov_mat @ natural_vec
        expec_covar = (expec_mean.unsqueeze(-1) @ expec_mean.unsqueeze(-2)).add_(cov_mat)
        return expec_mean, expec_covar

    @staticmethod
    def backward(ctx, expec_mean_grad, expec_covar_grad):
        return expec_mean_grad, expec_covar_grad


class NaturalVariationalDistribution(_VariationalDistribution):
    r"""
    Abstract base class for all Variational Distributions.
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        scaled_mean_init = torch.zeros(num_inducing_points)
        neg_prec_init = torch.eye(num_inducing_points, num_inducing_points).mul(-0.5)
        scaled_mean_init = scaled_mean_init.repeat(*batch_shape, 1)
        neg_prec_init = neg_prec_init.repeat(*batch_shape, 1, 1)

        # eta1 and eta2 parameterization of the variational distribution
        self.register_parameter(name="natural_vec", parameter=torch.nn.Parameter(scaled_mean_init))
        self.register_parameter(name="natural_mat", parameter=torch.nn.Parameter(neg_prec_init))

    def forward(self):
        r"""
        Constructs and returns the variational distribution

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:q(\mathbf u)"
        """

        # First - get the natural/canonical parameters (\theta in Hensman, 2013, eqn. 6)
        # This is what's computed by super().forward()
        # natural_vec = S^{-1} m
        # natural_mat = -1/2 S^{-1}
        natural_vec = self.natural_vec
        natural_mat = self.natural_mat

        # From the canonical parameters, compute the expectation parameters (\eta in Hensman, 2013, eqn. 6)
        # expec_mean = m = -1/2 natural_mat^{-1} natural_vec
        # expec_covar = m m^T + S = [ expec_mean expec_mean^T - 0.5 natural_mat^{-1} ]
        expec_mean, expec_covar = _ExpectationParams().apply(natural_vec, natural_mat)

        # Finally, convert the expected parameters into m and S
        # m = expec_mean
        # S = expec_covar - expec_mean expec_mean^T
        mean = expec_mean
        chol_covar = psd_safe_cholesky(expec_covar - expec_mean.unsqueeze(-1) @ expec_mean.unsqueeze(-2), max_tries=4)
        return MultivariateNormal(mean, CholLazyTensor(chol_covar))

    def initialize_variational_distribution(self, prior_dist):
        prior_prec = prior_dist.covariance_matrix.inverse()
        prior_mean = prior_dist.mean
        noise = torch.randn_like(prior_mean).mul_(self.mean_init_std)

        self.natural_vec.data.copy_((prior_prec @ prior_mean).add_(noise))
        self.natural_mat.data.copy_(prior_prec.mul(-0.5))
