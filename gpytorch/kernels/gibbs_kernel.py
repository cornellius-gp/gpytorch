#!/usr/bin/env python3

import math

import torch

from .kernel import Kernel


class GibbsKernel(Kernel):
    r"""Computes a covariance matrix based on the gibbs kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    """

    has_lengthscale = True

    def __init__(self, ard_num_dims: int = 1, batch_shape: torch.Size = torch.Size([]), **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        self._set_period_length(value)

    def _set_period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_period_length)
        self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.period_length).mul(math.pi)
        x2_ = x2.div(self.period_length).mul(math.pi)
        diff = x1_.unsqueeze(-2) - x2_.unsqueeze(-3)
        res = diff.sin().pow(2).sum(dim=-1).div(self.lengthscale).mul(-2.0).exp_()
        if diag:
            res = res.squeeze(0)
        return res
