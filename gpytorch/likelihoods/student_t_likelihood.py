#!/usr/bin/env python3

import torch

from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class StudentTLikelihood(_OneDimensionalLikelihood):
    r"""
    A Student T likelihood/noise model for GP regression.
    It has two learnable parameters: :math:`\nu` - the degrees of freedom, and
    :math:`\sigma^2` - the noise

    :var torch.Tensor deg_free: :math:`\nu` parameter (degrees of freedom)
    :var torch.Tensor noise: :math:`\sigma^2` parameter (noise)
    """

    def __init__(self, batch_shape=torch.Size([])):
        super().__init__()
        self._deg_free = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self._raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

    @property
    def deg_free(self):
        return torch.nn.functional.softplus(self._deg_free)

    @property
    def noise(self):
        return torch.nn.functional.softplus(self._raw_noise)

    def forward(self, function_samples, **kwargs):
        return base_distributions.StudentT(df=self.deg_free, loc=function_samples, scale=self.noise.sqrt())
