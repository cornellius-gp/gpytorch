import torch
from torch import nn
from .kernel import Kernel


class RBFKernel(Kernel):
    def __init__(self, log_lengthscale_bounds=(-10000, 10000), eps=1e-5):
        super(RBFKernel, self).__init__()
        self.eps = eps
        self.register_parameter('log_lengthscale', nn.Parameter(torch.zeros(1, 1)),
                                bounds=log_lengthscale_bounds)

    def forward(self, x1, x2):
        res = 2 * x1.matmul(x2.transpose(-1, -2))

        x1_squared = torch.matmul(x1.unsqueeze(-2), x1.unsqueeze(-1)).squeeze(-1)
        x2_squared = torch.matmul(x2.unsqueeze(-2), x2.unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
        res.sub_(x1_squared).sub_(x2_squared)  # res = -(x - z)^2

        res = res / (self.log_lengthscale.exp() + self.eps)  # res = -(x - z)^2 / lengthscale
        res.exp_()

        return res
