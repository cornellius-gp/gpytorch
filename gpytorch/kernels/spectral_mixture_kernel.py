import torch
import math
from torch.nn import Parameter
from torch.autograd import Function, Variable
from .kernel import Kernel

class SpectralMixtureKernel(Kernel):
    def forward(self, x1, x2, log_mixture_weights, log_mixture_means, log_mixture_scales):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError('The spectral mixture kernel can only be applied to a single dimension at a time. To use on multi-dimensional data, use a product of SM kernels, one for each dimension.')

        mixture_weights = log_mixture_weights.exp()
        mixture_means = log_mixture_means.exp()
        mixture_scales = log_mixture_scales.mul(2).exp_()

        sq_distance = torch.mm(x1, x2.transpose(0,1)).mul_(2)
        #sq_distance.addmm_(1, 2, x1, x2.transpose(0, 1)) # sq_distance = 2 x1 x2^T

        x1_squared = torch.bmm(x1.view(n, 1, d), x1.view(n, d, 1))
        x1_squared = x1_squared.view(n, 1).expand(n, m)
        x2_squared = torch.bmm(x2.view(m, 1, d), x2.view(m, d, 1))
        x2_squared = x2_squared.view(1, m).expand(n, m)

        sq_distance.add_(-x1_squared).add_(-x2_squared) # sq_distance = -(x - z)^2

        distance = torch.sqrt(sq_distance.mul(-1)) # distance = (x-z)

        sq_distance.mul_(2*math.pi**2) # sq_distance = -2*pi^2*(x-z)^2

        res = torch.zeros(n,m)
        for weight, mean, scale in zip(mixture_weights,mixture_means,mixture_scales):
            weight = weight.expand(n,m)
            mean = mean.expand(n,m)
            scale = scale.expand(n,m)

            sq_distance_factor = (scale * sq_distance).exp_()
            cos_factor = torch.cos(2*math.pi*mean.mul(distance))
            res = weight * sq_distance_factor * torch.cos(2*math.pi* mean * distance) # res += w_a^2*exp{-2\pi^2*\sigma_a^2*sq_distance}*cos(2\pi*\mu_a*distance)

        return res