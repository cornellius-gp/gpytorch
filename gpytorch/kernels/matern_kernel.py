import math
import torch
from torch import nn
from gpytorch.kernels import Kernel


class MaternKernel(Kernel):
    def __init__(self, nu, log_lengthscale_bounds=(-10000, 10000)):
        super(MaternKernel, self).__init__()
        if nu not in [0.5, 1.5, 2.5]:
            raise RuntimeError('nu expected to be 0.5, 1.5, or 2.5')
        self.nu = nu
        self.register_parameter('log_lengthscale', nn.Parameter(torch.zeros(1, 1, 1)), bounds=log_lengthscale_bounds)

    def forward(self, x1, x2):
        lengthscale = (self.log_lengthscale.exp()).sqrt()
        mean = x1.mean(1).mean(0)
        x1_normed = (x1 - mean.unsqueeze(0).unsqueeze(1)).div(lengthscale)
        x2_normed = (x2 - mean.unsqueeze(0).unsqueeze(1)).div(lengthscale)

        x1_squared = x1_normed.norm(2, -1).pow(2)
        x2_squared = x2_normed.norm(2, -1).pow(2)
        x1_t_x_2 = torch.matmul(x1_normed, x2_normed.transpose(-1, -2))

        distance_over_rho = (x1_squared.unsqueeze(-1) + x2_squared.unsqueeze(-2) - x1_t_x_2.mul(2))
        distance_over_rho = distance_over_rho.clamp(0, 1e10).sqrt()
        exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance_over_rho)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = ((math.sqrt(3) * distance_over_rho).add(1))
        elif self.nu == 2.5:
            constant_component = ((math.sqrt(5) * distance_over_rho).
                                  add(1).
                                  add(5. / 3. * distance_over_rho ** 2))

        return constant_component * exp_component
