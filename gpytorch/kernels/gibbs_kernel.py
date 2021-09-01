#!/usr/bin/env python3

import math

import torch

from ..constraints import Positive
from .kernel import Kernel


class GibbsKernel(Kernel):
    r"""Computes a covariance matrix based on the gibbs kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
    """

    has_lengthscale = True

    def __init__(self, period_length_prior=None, period_length_constraint=None, **kwargs):
        super(GibbsKernel, self).__init__(**kwargs)
        if period_length_constraint is None:
            period_length_constraint = Positive()

        self.register_parameter(
            name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        if period_length_prior is not None:
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, v: m._set_period_length(v),
            )

        self.register_constraint("raw_period_length", period_length_constraint)

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
