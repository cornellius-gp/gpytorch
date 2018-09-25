from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC

from torch.distributions import Distribution
from torch.nn import Module


class Prior(Distribution, Module, ABC):
    """
    Base class for Priors in GPyTorch.
    In GPyTorch, a parameter can be assigned a prior by passing it as the `prior` argument to
    :func:`~gpytorch.module.register_parameter`. GPyTorch performs internal bookkeeping of priors,
    and for each parameter with a registered prior includes the log probability of the parameter under its
    respective prior in computing the Marginal Log-Likelihood.
    """

    @property
    def log_transform(self):
        return self._log_transform

    def log_prob(self, parameter):
        """Returns the log-probability of the parameter value under the prior."""
        return super(Prior, self).log_prob(parameter.exp() if self.log_transform else parameter)
