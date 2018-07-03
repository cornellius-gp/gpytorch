from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.priors._compatibility import _bounds_to_prior


class PeriodicKernel(Kernel):
    def __init__(
        self,
        log_lengthscale_prior=None,
        log_period_length_prior=None,
        eps=1e-5,
        active_dims=None,
        log_lengthscale_bounds=(-10000, 10000),
        log_period_length_bounds=(-10000, 10000),
    ):
        log_lengthscale_prior = _bounds_to_prior(
            prior=log_lengthscale_prior, bounds=log_lengthscale_bounds
        )
        log_period_length_prior = _bounds_to_prior(
            prior=log_period_length_prior, bounds=log_period_length_bounds
        )
        super(PeriodicKernel, self).__init__(
            has_lengthscale=True,
            active_dims=active_dims,
            log_lengthscale_prior=log_lengthscale_prior,
        )
        self.eps = eps
        self.register_parameter(
            name="log_period_length",
            parameter=torch.nn.Parameter(torch.zeros(1, 1, 1)),
            prior=log_period_length_prior,
        )

    def forward(self, x1, x2):
        lengthscale = (self.log_lengthscale.exp() + self.eps).sqrt_()
        period_length = (self.log_period_length.exp() + self.eps).sqrt_()
        diff = torch.sum((x1.unsqueeze(2) - x2.unsqueeze(1)).abs(), -1)
        res = -2 * torch.sin(math.pi * diff / period_length).pow(2) / lengthscale
        return res.exp()
