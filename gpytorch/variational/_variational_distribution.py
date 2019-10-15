#!/usr/bin/env python3

import warnings
import torch
from ..module import Module
from abc import ABC, abstractmethod


class _VariationalDistribution(Module, ABC):
    """
    _VariationalDistribution objects represent the variational distribution q(u) over a set of inducing points for GPs.
    Calling it returns the variational distribution

    Args:
        num_inducing_points (int): Size of the variational distribution. This implies that the variational mean
            should be this size, and the variational covariance matrix should have this many rows and columns.
        batch_shape (torch.Size, optional): Specifies an optional batch
            size for the variational parameters. This is useful for example
            when doing additive variational inference.
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([])):
        super().__init__()

    def forward(self):
        """
        Constructs and returns the variational distribution

        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(u)
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_variational_distribution(self, prior_dist):
        """
        Method for initializing the variational distribution, based on the prior distribution.

        Args:
            :attr:`prior_dist` (gpytorch.distribution.Distribution):
                The prior distribution p(u)
        """
        raise NotImplementedError

    def __call__(self):
        try:
            return self.forward()
        # Deprecation added for 0.4 release
        except NotImplementedError:
            warnings.warn(
                "_VariationalDistribution.variational_distribution is deprecated. "
                "Please implement a `forward` method instead.",
                DeprecationWarning
            )
            return self.variational_distribution

    # Deprecation added for 0.4 release
    def __getattr__(self, attr):
        if attr == "variational_distribution":
            warnings.warn(
                "_VariationalDistribution.variational_distribution is deprecated. "
                "To get q(u), call the _VariationalDistribution object instead.",
                DeprecationWarning
            )
            return self.forward()
        else:
            return super().__getattr__(attr)
