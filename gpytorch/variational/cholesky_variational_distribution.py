#!/usr/bin/env python3

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from ._variational_distribution import _VariationalDistribution


class CholeskyVariationalDistribution(_VariationalDistribution):
    """
    VariationalDistribution objects represent the variational distribution q(u) over a set of inducing points for GPs.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower
    triangle.

    Args:
        :attr:`num_inducing_points` (int):
            Size of the variational distribution. This implies that the variational mean
            should be this size, and the variational covariance matrix should have this many rows and columns.
        :attr:`batch_shape` (torch.Size, optional):
            Specifies an optional batch size for the variational parameters. This is useful for example
            when doing additive variational inference.
        :attr:`mean_init_std` (float, default=1e-3):
            Standard deviation of gaussian noise to add to the mean initialization
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape)
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1, 1)

        self.mean_init_std = mean_init_std
        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", parameter=torch.nn.Parameter(covar_init))

    def forward(self):
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = chol_variational_covar.mul(lower_mask)

        # Now construct the actual matrix
        variational_covar = CholLazyTensor(chol_variational_covar)
        return MultivariateNormal(self.variational_mean, variational_covar)

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(self.mean_init_std, torch.randn_like(prior_dist.mean))
        self.chol_variational_covar.data.copy_(prior_dist.lazy_covariance_matrix.cholesky().evaluate())
