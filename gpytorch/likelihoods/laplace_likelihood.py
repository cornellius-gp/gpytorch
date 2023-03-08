#!/usr/bin/env python3

from typing import Any, Optional

import torch
from torch import Tensor
from torch.distributions import Laplace

from ..constraints import Interval, Positive
from ..distributions import base_distributions
from ..priors import Prior
from .likelihood import _OneDimensionalLikelihood


class LaplaceLikelihood(_OneDimensionalLikelihood):
    r"""
    A Laplace likelihood/noise model for GP regression.
    It has one learnable parameter: :math:`\sigma` - the noise

    :param batch_shape: The batch shape of the learned noise parameter (default: []).
    :param noise_prior: Prior for noise parameter :math:`\sigma`.
    :param noise_constraint: Constraint for noise parameter :math:`\sigma`.

    :var torch.Tensor noise: :math:`\sigma` parameter (noise)
    """

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        noise_prior: Optional[Prior] = None,
        noise_constraint: Optional[Interval] = None,
    ) -> None:
        super().__init__()

        if noise_constraint is None:
            noise_constraint = Positive()

        self.raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda m: m.noise, lambda m, v: m._set_noise(v))

        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self) -> Tensor:
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self._set_noise(value)

    def _set_noise(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Laplace:
        return base_distributions.Laplace(loc=function_samples, scale=self.noise.sqrt())
