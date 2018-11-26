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

    # convert from normal expectations eta=(eta1, eta2) to mu, L representation
    def _dist_from_natural(self, nat_mean, nat_L):
        mumu = torch.matmul(nat_mean.unsqueeze(-1), nat_mean.unsqueeze(-2))
        L = torch.cholesky(nat_L.matmul(nat_L.transpose(-2, -1)) - mumu, upper=False)
        return nat_mean, L.contiguous()

    # convert to eta representation from mu, L representation
    def _natural_from_dist(self, mu, Sigma):
        mumu = torch.matmul(mu.unsqueeze(-1), mu.unsqueeze(-2))
        nat_L = torch.cholesky(Sigma + mumu, upper=False)
        return mu, nat_L

    # Initialize eta1 and eta2 from prior distribution via _natural_from_dist
    def initialize_variational_distribution(self, prior_dist):
        prior_mean = prior_dist.mean
        prior_covar = prior_dist.covariance_matrix

        nat_mean, nat_L = self._natural_from_dist(prior_mean, prior_covar)

        self.natural_variational_mean.data.copy_(nat_mean)
        self.natural_variational_covar.data.copy_(nat_L)

    @property
    def variational_distribution(self):
        if self.has_buffer:
            # The idea is to cache mu and L for each optimization step, then clear them in step().
            # This way, we can use torch.autograd.grad calls on the loss.
            variational_mean, chol_variational_covar = self.buffer
        else:
            variational_mean, chol_variational_covar = self._dist_from_natural(
                self.natural_variational_mean,
                self.natural_variational_covar
            )
            self.buffer = (variational_mean, chol_variational_covar)
            self.has_buffer = True

        variational_covar = CholLazyTensor(chol_variational_covar)
        return MultivariateNormal(variational_mean, variational_covar)
