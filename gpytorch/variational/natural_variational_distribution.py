#!/usr/bin/env python3

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from .variational_distribution import VariationalDistribution


class NaturalVariationalDistribution(VariationalDistribution):
    """
    VariationalDistribution objects represent the variational distribution q(u) over a set of inducing points for GPs.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower triangle
    and we manually ensure that the diagonal remains positive.
    """

    def __init__(self, num_inducing_points, batch_size=None):
        """
        Args:
            num_inducing_points (int): Size of the variational distribution. This implies that the variational mean
                should be this size, and the variational covariance matrix should have this many rows and columns.
            batch_size (int, optional): Specifies an optional batch size for the variational parameters. This is useful
                for example when doing additive variational inference.
        """
        super(VariationalDistribution, self).__init__()
        mean_init = 0.03 * torch.randn(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        if batch_size is not None:
            mean_init = mean_init.repeat(batch_size, 1)
            covar_init = covar_init.repeat(batch_size, 1, 1)

        # eta1 and eta2 parameterization of the variational distribution
        self.register_parameter(name="natural_variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="natural_variational_covar", parameter=torch.nn.Parameter(covar_init))

        # Lets us buffer mu and L after computing from eta1 and eta2 so that the optimizer has access
        # to the right variables. This gets cleared by NaturalAdam.step
        self.has_buffer = False

    def _meanvarsqrt_to_expectation(self, mean, chol_covar):
        v = torch.matmul(chol_covar, torch.transpose(chol_covar, -1, -2))
        return mean, v + torch.matmul(mean.unsqueeze(-1), mean.unsqueeze(-2))

    def _meanvarsqrt_to_natural(self, mean, chol_covar):
        chol_covar_inv = torch.trtrs(torch.eye(mean.size(-1)).type_as(mean), chol_covar, upper=False)[0]
        s_inv = torch.matmul(torch.transpose(chol_covar_inv, -1, -2), chol_covar_inv)
        return torch.matmul(s_inv, mean), -0.5 * s_inv

    def _expectation_to_meanvarsqrt(self, eta_1, eta_2):
        var = eta_2 - torch.matmul(eta_1.unsqueeze(-1), eta_1.unsqueeze(-2))
        return eta_1, torch.cholesky(var, upper=False)

    def _natural_to_meanvarsqrt(self, nat_mean, nat_covar):
        var_sqrt_inv = torch.cholesky(-2.0 * nat_covar, upper=False)
        var_sqrt = torch.trtrs(torch.eye(nat_mean.size(-1)).type_as(nat_mean), var_sqrt_inv, upper=False)[0]
        # Probably could just return var_sqrt here, and do mu = CholLazyTensor(var_sqrt).matmul(nat_mean)
        S = torch.matmul(torch.transpose(var_sqrt, -1, -2), var_sqrt)
        mu = torch.matmul(S, nat_mean)
        return mu, torch.cholesky(S, upper=False)

    # Initialize eta1 and eta2 from prior distribution via _natural_from_dist
    def initialize_variational_distribution(self, prior_dist):
        prior_mean = prior_dist.mean
        prior_covar_sqrt = torch.cholesky(prior_dist.covariance_matrix, upper=False)

        nat_mean, nat_covar = self._meanvarsqrt_to_natural(prior_mean.detach(), prior_covar_sqrt.detach())

        self.natural_variational_mean.data.copy_(nat_mean)
        self.natural_variational_covar.data.copy_(nat_covar)

    @property
    def variational_distribution(self):
        if self.has_buffer:
            # The idea is to cache mu and L for each optimization step, then clear them in step().
            # This way, we can use torch.autograd.grad calls on the loss.
            variational_mean, chol_variational_covar = self.buffer
        else:
            variational_mean, chol_variational_covar = self._natural_to_meanvarsqrt(
                self.natural_variational_mean,
                self.natural_variational_covar
            )
            self.buffer = (variational_mean, chol_variational_covar)
            self.has_buffer = True

        variational_covar = CholLazyTensor(chol_variational_covar)
        return MultivariateNormal(variational_mean, variational_covar)
