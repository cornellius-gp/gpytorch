#!/usr/bin/env python3

import warnings
from abc import ABC, abstractmethod

import torch

from ..module import Module


class _VariationalDistribution(Module, ABC):
    r"""
    Abstract base class for all Variational Distributions.

    :ivar torch.dtype dtype: The dtype of the VariationalDistribution parameters
    :ivar torch.dtype device: The device of the VariationalDistribution parameters
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3):
        super().__init__()
        self.num_inducing_points = num_inducing_points
        self.batch_shape = batch_shape
        self.mean_init_std = mean_init_std

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self):
        r"""
        Constructs and returns the variational distribution

        :rtype: ~gpytorch.distributions.MultivariateNormal
        :return: The distribution :math:`q(\mathbf u)`
        """
        raise NotImplementedError

    def shape(self) -> torch.Size:
        r"""
        Event + batch shape of VariationalDistribution object
        :rtype: torch.Size
        """
        return torch.Size([*self.batch_shape, self.num_inducing_points])

    @abstractmethod
    def initialize_variational_distribution(self, prior_dist):
        r"""
        Method for initializing the variational distribution, based on the prior distribution.

        :param ~gpytorch.distributions.Distribution prior_dist: The prior distribution :math:`p(\mathbf u)`.
        """
        raise NotImplementedError

    def __call__(self):
        try:
            return self.forward()
        # Remove after 1.0
        except NotImplementedError:
            warnings.warn(
                "_VariationalDistribution.variational_distribution is deprecated. "
                "Please implement a `forward` method instead.",
                DeprecationWarning,
            )
            return self.variational_distribution

    def __getattr__(self, attr):
        # Remove after 1.0
        if attr == "variational_distribution":
            warnings.warn(
                "_VariationalDistribution.variational_distribution is deprecated. "
                "To get q(u), call the _VariationalDistribution object instead.",
                DeprecationWarning,
            )
            return self.forward()
        else:
            return super().__getattr__(attr)
