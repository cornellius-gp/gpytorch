#!/usr/bin/env python3

import torch
from ..functions import add_diag
from ..lazy import CholLazyTensor, NonLazyTensor
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
        self.chol_variational_covar.data.copy_(torch.cholesky(prior_dist.covariance_matrix, upper=True))

    @property
    def variational_distribution(self):
        """
        Return the variational distribution q(u) that this module represents.

        In this simplest case, this involves directly returning the variational mean. For the variational covariance
        matrix, we consider the lower triangle of the registered variational covariance parameter, while also ensuring
        that the diagonal remains positive.
        """
        chol_variational_covar = self.chol_variational_covar
        diagonal = NonLazyTensor(chol_variational_covar).diag()
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        # And has a positive diagonal
        strictly_lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).triu(1)
        diagonal = torch.nn.functional.softplus(diagonal)
        chol_variational_covar = add_diag(chol_variational_covar.mul(strictly_lower_mask), diagonal)

        # Now construct the actual matrix
        variational_covar = CholLazyTensor(chol_variational_covar.transpose(-1, -2))
        return MultivariateNormal(self.variational_mean, variational_covar)
