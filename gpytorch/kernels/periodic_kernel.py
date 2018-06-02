from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch
from torch import nn
from .kernel import Kernel


class PeriodicKernel(Kernel):
    def __init__(
        self,
        log_lengthscale_bounds=(-10000, 10000),
        log_period_length_bounds=(-10000, 10000),
        eps=1e-5,
        active_dims=None,
    ):
        super(PeriodicKernel, self).__init__(
            has_lengthscale=True, log_lengthscale_bounds=log_lengthscale_bounds, active_dims=active_dims
        )
        self.eps = eps
        self.register_parameter(
            "log_period_length", nn.Parameter(torch.zeros(1, 1, 1)), bounds=log_period_length_bounds
        )

    def forward(self, x1, x2):
        lengthscale = (self.log_lengthscale.exp() + self.eps).sqrt_()
        period_length = (self.log_period_length.exp() + self.eps).sqrt_()
        diff = torch.sum((x1.unsqueeze(2) - x2.unsqueeze(1)).abs(), -1)
        res = -2 * torch.sin(math.pi * diff / period_length).pow(2) / lengthscale
        return res.exp()
