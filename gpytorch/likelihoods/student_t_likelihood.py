#!/usr/bin/env python3

import torch

from ..constraints import GreaterThan, Positive
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

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        deg_free_prior=None,
        deg_free_constraint=None,
        noise_prior=None,
        noise_constraint=None,
    ):
        super().__init__()

        if deg_free_constraint is None:
            deg_free_constraint = GreaterThan(2)

        if noise_constraint is None:
            noise_constraint = Positive()

        self.raw_deg_free = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda: self.noise, lambda v: self._set_noise(v))

        self.register_constraint("raw_noise", noise_constraint)

        if deg_free_prior is not None:
            self.register_prior(
                "deg_free_prior", deg_free_prior, lambda: self.deg_free, lambda v: self._set_deg_free(v)
            )

        self.register_constraint("raw_deg_free", deg_free_constraint)

        # Rough initialization
        self.initialize(deg_free=7)

    @property
    def deg_free(self):
        return self.raw_deg_free_constraint.transform(self.raw_deg_free)

    @deg_free.setter
    def deg_free(self, value):
        self._set_deg_free(value)

    def _set_deg_free(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_deg_free)
        self.initialize(raw_deg_free=self.raw_deg_free_constraint.inverse_transform(value))

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
        return base_distributions.StudentT(df=self.deg_free, loc=function_samples, scale=self.noise.sqrt())
