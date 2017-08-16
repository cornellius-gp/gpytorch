import torch
import gpytorch
from torch import nn
from gpytorch.random_variables import GaussianRandomVariable
from .likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    def __init__(self, log_noise_bounds=(-1000, 1000)):
        super(GaussianLikelihood, self).__init__()
        self.register_parameter('log_noise', nn.Parameter(torch.zeros(1)), bounds=log_noise_bounds)

    def forward(self, input):
        assert(isinstance(input, GaussianRandomVariable))
        mean, covar = input.representation()
        noise = gpytorch.add_diag(covar, self.log_noise.exp())
        return GaussianRandomVariable(mean, noise)
