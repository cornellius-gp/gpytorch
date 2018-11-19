#!/usr/bin/env python3

from ..module import Module


class VariationalDistribution(Module):
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

    @property
    def variational_distribution(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise RuntimeError("VariationalDistribution is not intended to be called!")

    def initialize_variational_distribution(self, prior_dist):
        """
        Method for initializing the variational distribution, based on the prior distribution.

        Args:
            - `prior_dist` (gpytorch.distribution.Distribution): the prior distribution
        """
        raise NotImplementedError
