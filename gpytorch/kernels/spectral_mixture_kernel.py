import math
import torch
from torch import nn
from .kernel import Kernel
from torch.autograd import Variable
import gpytorch.utils.fft as fft


class SpectralMixtureKernel(Kernel):
    def __init__(self, n_mixtures, log_mixture_weight_bounds=(-100, 100),
                 log_mixture_mean_bounds=(-100, 100), log_mixture_scale_bounds=(-100, 100)):
        super(SpectralMixtureKernel, self).__init__()
        self.n_mixtures = n_mixtures
        self.register_parameter('log_mixture_weights', nn.Parameter(torch.zeros(n_mixtures)),
                                bounds=log_mixture_weight_bounds)
        self.register_parameter('log_mixture_means', nn.Parameter(torch.zeros(n_mixtures)),
                                bounds=log_mixture_mean_bounds)
        self.register_parameter('log_mixture_scales', nn.Parameter(torch.zeros(n_mixtures)),
                                bounds=log_mixture_scale_bounds)

    def initialize(self, x_train, y, **kwargs):
        if isinstance(x_train, Variable):
            x_train = x_train.data
        if isinstance(y, Variable):
            y = y.data

        x_train_sort = x_train.sort()[0]
        max_dis = x_train_sort[-1] - x_train_sort[0]
        min_dis = torch.min(x_train_sort[1:] - x_train_sort[:-1])

        scales = 1.0 / torch.abs(max_dis * torch.randn(self.n_mixtures))
        means = (0.5 / min_dis) * torch.rand(self.n_mixtures)
        weights = torch.ones(self.n_mixtures) * (y.std() / self.n_mixtures)

        self.log_mixture_weights.data = weights.log()
        self.log_mixture_scales.data = scales.log()
        self.log_mixture_means.data = means.log()

    def forward(self, x1, x2):
        n, d = x1.size()
        m, _ = x2.size()

        if d > 1:
            raise RuntimeError(' '.join([
                'The spectral mixture kernel can only be applied'
                'to a single dimension at a time. To use on multi-dimensional data,',
                'use a product of SM kernels, one for each dimension.'
            ]))

        mixture_weights = self.log_mixture_weights.exp()
        mixture_means = self.log_mixture_means.exp()
        mixture_scales = self.log_mixture_scales.mul(2).exp_()

        sq_distance = torch.mm(x1, x2.transpose(0, 1)).mul_(2)

        x1_squared = torch.bmm(x1.view(n, 1, d), x1.view(n, d, 1))
        x1_squared = x1_squared.view(n, 1).expand(n, m)
        x2_squared = torch.bmm(x2.view(m, 1, d), x2.view(m, d, 1))
        x2_squared = x2_squared.view(1, m).expand(n, m)

        sq_distance.add_(-x1_squared).add_(-x2_squared)  # sq_distance = -(x - z)^2

        distance = torch.sqrt(sq_distance.mul(-1))  # distance = (x-z)

        sq_distance.mul_(2 * math.pi ** 2)  # sq_distance = -2*pi^2*(x-z)^2

        res = torch.zeros(n, m)
        for weight, mean, scale in zip(mixture_weights, mixture_means, mixture_scales):
            weight = weight.expand(n, m)
            mean = mean.expand(n, m)
            scale = scale.expand(n, m)

            sq_distance_factor = (scale * sq_distance).exp_()
            res = weight * sq_distance_factor * torch.cos(2 * math.pi * mean * distance)
            # res += w_a^2*exp{-2\pi^2*\sigma_a^2*sq_distance}*cos(2\pi*\mu_a*distance)

        return res
