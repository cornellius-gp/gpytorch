#!/usr/bin/env python3

from abc import ABC

from ..distributions import Distribution
from torch.nn import Module


class Prior(Distribution, Module, ABC):
    """
    Base class for Priors in GPyTorch.
    In GPyTorch, a parameter can be assigned a prior by passing it as the `prior` argument to
    :func:`~gpytorch.module.register_parameter`. GPyTorch performs internal bookkeeping of priors,
    and for each parameter with a registered prior includes the log probability of the parameter under its
    respective prior in computing the Marginal Log-Likelihood.
    """

    def transform(self, x):
        return self._transform(x) if self._transform is not None else x

    def log_prob(self, x):
        """Returns the log-probability of the parameter value under the prior."""
        return super(Prior, self).log_prob(self.transform(x))
