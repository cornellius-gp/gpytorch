#!/usr/bin/env python3

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

    @property
    def variational_distribution(self):
        """
        Constructs and returns the variational distribution

        Returns:
            :obj:`gpytorch.distributions.MultivariateNormal`: The distribution q(u)
        """
        return self()

    @abstractmethod
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
        return self.forward()
