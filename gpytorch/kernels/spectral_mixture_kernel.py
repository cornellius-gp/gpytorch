import math
import torch
from torch import nn
from .kernel import Kernel


class SpectralMixtureKernel(Kernel):
    def __init__(self, n_mixtures, n_dims=1, log_mixture_weight_bounds=(-100, 100),
                 log_mixture_mean_bounds=(-100, 100), log_mixture_scale_bounds=(-100, 100)):
        self.n_mixtures = n_mixtures
        self.n_dims = n_dims

        super(SpectralMixtureKernel, self).__init__()
        self.register_parameter('log_mixture_weights', nn.Parameter(torch.zeros(self.n_mixtures)),
                                bounds=log_mixture_weight_bounds)
        self.register_parameter('log_mixture_means', nn.Parameter(torch.zeros(self.n_mixtures, self.n_dims)),
                                bounds=log_mixture_mean_bounds)
        self.register_parameter('log_mixture_scales', nn.Parameter(torch.zeros(self.n_mixtures, self.n_dims)),
                                bounds=log_mixture_scale_bounds)

    def forward(self, x1, x2):
        batch_size, n, n_dims = x1.size()
        _, m, _ = x2.size()
        if not n_dims == self.n_dims:
            raise RuntimeError('The number of dimensions doesn\'t match what was supplied!')

        mixture_weights = self.log_mixture_weights.exp().view(self.n_mixtures, 1, 1, 1)
        mixture_means = self.log_mixture_means.exp().view(self.n_mixtures, 1, 1, 1, self.n_dims)
        mixture_scales = self.log_mixture_scales.exp().view(self.n_mixtures, 1, 1, 1, self.n_dims)
        distance = (x1.unsqueeze(-2) - x2.unsqueeze(-3))  # distance = x^(i) - z^(i)

        exp_term = (distance * mixture_scales).pow_(2).mul_(-2 * math.pi ** 2)
        cos_term = (distance * mixture_means).mul_(2 * math.pi)
        res = exp_term.exp_() * cos_term.cos_()

        # Product over dimensions
        res = res.prod(-1)

        # Sum over mixtures
        res = (res * mixture_weights).sum(0)
        return res
