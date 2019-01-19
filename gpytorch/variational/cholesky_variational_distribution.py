#!/usr/bin/env python3

import torch
from ..lazy import CholLazyTensor
from ..distributions import MultivariateNormal
from .variational_distribution import VariationalDistribution


class CholeskyVariationalDistribution(VariationalDistribution):
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
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        if batch_size is not None:
            mean_init = mean_init.repeat(batch_size, 1)
            covar_init = covar_init.repeat(batch_size, 1, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", parameter=torch.nn.Parameter(covar_init))

    def initialize_variational_distribution(self, prior_dist):
        self.variational_mean.data.copy_(prior_dist.mean)
        self.chol_variational_covar.data.copy_(prior_dist.covariance_matrix.inverse().cholesky())

    @property
    def variational_distribution(self):
        """
        Return the variational distribution q(u) that this module represents.

        In this simplest case, this involves directly returning the variational mean. For the variational covariance
        matrix, we consider the lower triangle of the registered variational covariance parameter, while also ensuring
        that the diagonal remains positive.
        """
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = chol_variational_covar.mul(lower_mask)

        # Now construct the actual matrix
        variational_covar = CholLazyTensor(chol_variational_covar)
        return MultivariateNormal(self.variational_mean, variational_covar)
