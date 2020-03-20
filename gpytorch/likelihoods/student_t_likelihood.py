#!/usr/bin/env python3

import torch

from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class StudentTLikelihood(_OneDimensionalLikelihood):
    def __init__(self, deg_free=3.0):
        super().__init__()
        self.deg_free = deg_free
        self._raw_noise = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def noise(self):
        return torch.nn.functional.softplus(self._raw_noise)

    def forward(self, function_samples, **kwargs):
        return base_distributions.StudentT(df=self.deg_free, loc=function_samples, scale=self.noise.sqrt())
