#!/usr/bin/env python3

import torch

from ..distributions import base_distributions
from ..utils.transforms import inv_softplus
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
        self.raw_deg_free = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        # Rough initialization
        self.initialize(deg_free=7)

    @property
    def deg_free(self):
        return 2 + torch.nn.functional.softplus(self.raw_deg_free)

    @deg_free.setter
    def deg_free(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_deg_free)
        self.initialize(raw_deg_free=inv_softplus(value - 2))

    @property
    def noise(self):
        return torch.nn.functional.softplus(self.raw_noise)

    @noise.setter
    def noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=inv_softplus(value))

    def forward(self, function_samples, **kwargs):
        return base_distributions.StudentT(df=self.deg_free, loc=function_samples, scale=self.noise.sqrt())
