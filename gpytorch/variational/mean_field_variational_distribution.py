#!/usr/bin/env python3

import torch
from ..lazy import DiagLazyTensor
from ..distributions import MultivariateNormal
from .variational_distribution import VariationalDistribution
from ..utils.deprecation import _deprecate_kwarg_with_transform


class MeanFieldVariationalDistribution(VariationalDistribution):
    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), **kwargs):
        """
        Args:
            num_inducing_points (int): Size of the variational distribution. This implies that the variational mean
                should be this size, and the variational covariance matrix should have this many rows and columns.
            batch_shape (torch.Size, optional): Specifies an optional batch
                size for the variational parameters. This is useful for example
                when doing additive variational inference.
        """
        batch_shape = _deprecate_kwarg_with_transform(
            kwargs, "batch_size", "batch_shape", batch_shape, lambda n: torch.Size([n])
        )
        super(VariationalDistribution, self).__init__()
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.ones(num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="variational_var", parameter=torch.nn.Parameter(covar_init))

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.chol_variational_covar.data.copy_(prior_dist.variance)

    @property
    def variational_distribution(self):
        variational_var = self.variational_var
        dtype = variational_var.dtype
        device = variational_var.device

        # Now construct the actual matrix
        variational_covar = DiagLazyTensor(variational_var.pow(2))
        return MultivariateNormal(self.variational_mean, variational_covar)
