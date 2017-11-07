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
        n, d = x1.size()
        m, _ = x2.size()

        res = 2 * x1.matmul(x2.transpose(0, 1))

        x1_squared = torch.bmm(x1.view(n, 1, d), x1.view(n, d, 1))
        x1_squared = x1_squared.view(n, 1).expand(n, m)
        x2_squared = torch.bmm(x2.view(m, 1, d), x2.view(m, d, 1))
        x2_squared = x2_squared.view(1, m).expand(n, m)
        res.sub_(x1_squared).sub_(x2_squared)  # res = -(x - z)^2

        res = res / (self.log_lengthscale.exp() + self.eps)  # res = -(x - z)^2 / lengthscale
        res.exp_()
        return res
