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
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :type batch_shape: :obj:`torch.Size`, optional
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
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
        # TODO: if we don't multiply self._variational_stddev by a mask of one, Pyro models fail
        # not sure where this bug is occuring (in Pyro or PyTorch)
        # throwing this in as a hotfix for now - we should investigate later
        mask = torch.ones_like(self._variational_stddev)
        return self._variational_stddev.mul(mask).abs().clamp_min(1e-8)

    def forward(self):
        # TODO: if we don't multiply self._variational_stddev by a mask of one, Pyro models fail
        # not sure where this bug is occuring (in Pyro or PyTorch)
        # throwing this in as a hotfix for now - we should investigate later
        mask = torch.ones_like(self._variational_stddev)
        variational_covar = DiagLazyTensor(self._variational_stddev.mul(mask).pow(2))
        return MultivariateNormal(self.variational_mean, variational_covar)

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.variational_mean.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
        self._variational_stddev.data.copy_(prior_dist.stddev)
