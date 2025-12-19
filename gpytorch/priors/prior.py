#!/usr/bin/env python3

from abc import ABC
from typing import Any, Mapping

import torch

from torch.distributions import TransformedDistribution
from torch.nn import Module

from ..distributions import Distribution
from .utils import _load_transformed_to_base_dist


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
        # If setting a _transformed_ attribute, update the base attribute instead.
        # Note: The _transformed_ prefix indicates this attribute is a buffer
        # belonging to a TransformedDistribution. It is NOT a transformed version
        # of the value - rather, it's a direct copy of the base_dist attribute
        # (e.g., _transformed_loc mirrors base_dist.loc exactly).
        tensor_value = torch.as_tensor(value)
        if hasattr(self, name) and "_transformed_" in name:
            base_attr_name = name.replace("_transformed_", "")
            # Update the base attribute in the base distribution
            self.base_dist.__setattr__(base_attr_name, tensor_value)
            # Update the buffer copy as well (must stay in sync with base_dist)
            super().__setattr__(name, tensor_value)

        elif hasattr(self, f"_transformed_{name}"):
            self.base_dist.__setattr__(name, tensor_value)
            super().__setattr__(f"_transformed_{name}", tensor_value)

        else:
            return super().__setattr__(name, value)
