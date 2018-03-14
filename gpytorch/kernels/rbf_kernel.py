from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
from .kernel import Kernel


class RBFKernel(Kernel):
    def __init__(
        self,
        log_lengthscale_bounds=(-10000, 10000),
        eps=1e-5,
        active_dims=None,
    ):
        super(RBFKernel, self).__init__(active_dims=active_dims)
        self.eps = eps
        self.register_parameter('log_lengthscale', nn.Parameter(torch.zeros(1, 1)),
                                bounds=log_lengthscale_bounds)

    def forward(self, x1, x2):
        lengthscale = (self.log_lengthscale.exp() + self.eps).sqrt_()
        mean = x1.mean(1).mean(0)
        x1_normed = (x1 - mean.unsqueeze(0).unsqueeze(1)).div_(lengthscale)
        x2_normed = (x2 - mean.unsqueeze(0).unsqueeze(1)).div_(lengthscale)

        x1_squared = x1_normed.norm(2, -1).pow(2)
        x2_squared = x2_normed.norm(2, -1).pow(2)
        x1_t_x_2 = torch.matmul(x1_normed, x2_normed.transpose(-1, -2))
        res = (x1_squared.unsqueeze(-1) - x1_t_x_2.mul_(2) + x2_squared.unsqueeze(-2)).mul_(-1)
        res = res.exp()
        return res
