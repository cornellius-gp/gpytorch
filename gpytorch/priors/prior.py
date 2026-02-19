#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from typing import Any

import torch
from torch.distributions import TransformedDistribution
from torch.nn import Module

from ..distributions import Distribution
from .utils import _load_transformed_to_base_dist, BUFFERED_PREFIX


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
        return super().log_prob(self.transform(x))

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        Module.load_state_dict(self, state_dict, *args, **kwargs)
        if isinstance(self, TransformedDistribution):
            _load_transformed_to_base_dist(self)

    def __setattr__(self, name: str, value: Any) -> None:
        # If setting a BUFFERED_PREFIX attribute, update the base attribute instead.
        # Note: BUFFERED_PREFIX is just an indicator that this attribute belongs to a
        # TransformedDistribution, the value itself is not transformed.
        if hasattr(self, name) and BUFFERED_PREFIX in name:
            base_attr_name = name.replace(BUFFERED_PREFIX, "")
            # Convert to Tensor if needed
            tensor_value = torch.as_tensor(value)
            # Update the base attribute in the base distribution
            self.base_dist.__setattr__(base_attr_name, tensor_value)
            # Update the transformed attribute as well
            super().__setattr__(name, tensor_value)

        elif hasattr(self, f"{BUFFERED_PREFIX}{name}"):
            self.base_dist.__setattr__(name, value)
            super().__setattr__(f"{BUFFERED_PREFIX}{name}", value)

        else:
            return super().__setattr__(name, value)
