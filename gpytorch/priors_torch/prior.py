from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC

from torch.distributions import Distribution
from torch.nn import Module


class Prior(Distribution, Module, ABC):
    """Priors ar imposed on hyperparameters"""

    @property
    def log_transform(self):
        return self._log_transform

    def log_prob(self, parameter):
        """Returns the log-probability of the parameter value under the prior."""
        return super(Prior, self).log_prob(parameter.exp() if self.log_transform else parameter)
