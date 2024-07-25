#!/usr/bin/env python3

from abc import ABC
from typing import Any, Mapping

from torch.distributions import TransformedDistribution
from torch.nn import Module

from ..distributions import Distribution
from .utils import _load_transformed_to_base_dist


TRANSFORMED_ERROR_MSG = """Priors of TransformedDistributions should not have their \
'_transformed' attributes modified, these are just copies of the base attribute. \
Please modify the base attribute (e.g. {}) instead."""


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
        r"""
        :return: log-probability of the parameter value under the prior
        :rtype: torch.Tensor
        """
        return super(Prior, self).log_prob(self.transform(x))

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        Module.load_state_dict(self, state_dict, *args, **kwargs)
        if isinstance(self, TransformedDistribution):
            _load_transformed_to_base_dist(self)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, name) and "_transformed_" in name:
            base_attr_name = name.replace("_transformed_", "")
            raise AttributeError(TRANSFORMED_ERROR_MSG.format(base_attr_name))

        elif hasattr(self, f"_transformed_{name}"):
            self.base_dist.__setattr__(name, value)
            super().__setattr__(f"_transformed_{name}", value)

        else:
            return super().__setattr__(name, value)
