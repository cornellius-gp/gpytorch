from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import abstractmethod, abstractproperty
import torch
from torch.nn import Module


class Prior(Module):
    """
    Base class for Priors in GPyTorch.
    In GPyTorch, a parameter can be assigned a prior by passing it as the `prior` argument to
    :func:`~gpytorch.module.register_parameter`. GPyTorch performs internal bookkeeping of priors,
    and for each parameter with a registered prior includes the log probability of the parameter under its
    respective prior in computing the Marginal Log-Likelihood (see e.g. :func:`~gpytorch.priors.Prior.forward`).

    In order to define a new Prior in GPyTorch, a user must define at a minimum the following methods
    * :func:`~gpytorch.priors.Prior.shape`, which returns the shape of the domain as a torch.Size.
    * :func:`~gpytorch.priors.Prior.is_in_support`, which for a given parameter value returns a bool indicating
        whether the parameter is contained in the support of the distribution.
    * :func:`~gpytorch.priors.Prior._log_prob`, which returns the log-probability for a given parameter value
        as a torch scalar.

    """

    @abstractmethod
    def is_in_support(self, parameter):
        raise NotImplementedError()

    @property
    def log_transform(self):
        return self._log_transform

    @abstractmethod
    def _log_prob(self, parameter):
        raise NotImplementedError()

    @abstractproperty
    def shape(self):
        raise NotImplementedError()

    def log_prob(self, parameter):
        """Returns the log-probability of the parameter value under the prior."""
        return self._log_prob(parameter.exp() if self.log_transform else parameter)

    def _initialize_distributions(self):
        pass

    def _apply(self, fn):
        Module._apply(self, fn)
        self._initialize_distributions()


class TorchDistributionPrior(Prior):
    """
    Base class for Priors based on pytorch Distribution objects.
    """

    def _log_prob(self, parameter):
        return sum(
            d.log_prob(p)
            for d, p in zip(self._distributions, parameter.view(*self.shape))
        )

    @property
    def shape(self):
        return torch.Size([len(self._distributions)])
