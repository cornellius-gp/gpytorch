#!/usr/bin/env python3

import torch

from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class LaplaceLikelihood(_OneDimensionalLikelihood):
    r"""
    A Laplace likelihood/noise model for GP regression.
    It has one learnable parameter: :math:`\sigma^2` - the noise

    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(self, batch_shape=torch.Size([])):
        super().__init__()
        self._raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

    @property
    def noise(self):
        return torch.nn.functional.softplus(self._raw_noise)

    def forward(self, function_samples, **kwargs):
        return base_distributions.Laplace(loc=function_samples, scale=self.noise.sqrt())
