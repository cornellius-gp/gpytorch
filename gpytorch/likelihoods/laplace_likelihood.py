#!/usr/bin/env python3

import torch

from ..constraints import Positive
from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class LaplaceLikelihood(_OneDimensionalLikelihood):
    r"""
    A Laplace likelihood/noise model for GP regression.
    It has one learnable parameter: :math:`\sigma^2` - the noise

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(self, batch_shape=torch.Size([]), noise_prior=None, noise_constraint=None):
        super().__init__()

        if noise_constraint is None:
            noise_constraint = Positive()

        self.raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda: self.noise, lambda v: self._set_noise(v))

        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(self, function_samples, **kwargs):
        return base_distributions.Laplace(loc=function_samples, scale=self.noise.sqrt())
