#!/usr/bin/env python3

import torch

from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor
from ._variational_distribution import _VariationalDistribution


class MeanFieldVariationalDistribution(_VariationalDistribution):
    """
    A :obj:`~gpytorch.variational._VariationalDistribution` that is defined to be a multivariate normal distribution
    with a diagonal covariance matrix. This will not be as flexible/expressive as a
    :obj:`~gpytorch.variational.CholeskyVariationalDistribution`.

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param torch.Size batch_shape: (Optional.) Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :param float mean_init_std: (default=1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.ones(num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="_variational_stddev", parameter=torch.nn.Parameter(covar_init))

    @property
    def variational_stddev(self):
        return self._variational_stddev.abs().clamp_min(1e-8)

    def forward(self):
        variational_var = self.variational_stddev.pow(2)
        variational_covar = DiagLazyTensor(variational_var)
        return MultivariateNormal(self.variational_mean, variational_covar)

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(self.mean_init_std, torch.randn_like(prior_dist.mean))
        self.variational_stddev.data.copy_(prior_dist.stddev)
