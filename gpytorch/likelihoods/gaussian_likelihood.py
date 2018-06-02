from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import math
from torch import nn
from ..functions import add_diag
from ..random_variables import GaussianRandomVariable
from .likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    def __init__(self, log_noise_bounds=(-1000, 1000)):
        super(GaussianLikelihood, self).__init__()
        self.register_parameter("log_noise", nn.Parameter(torch.zeros(1)), bounds=log_noise_bounds)

    def forward(self, input):
        assert isinstance(input, GaussianRandomVariable)
        mean, covar = input.representation()
        noise = add_diag(covar, self.log_noise.exp())
        return GaussianRandomVariable(mean, noise)

    def log_probability(self, input, target):
        res = -0.5 * ((target - input.mean()) ** 2 + input.var()) / self.log_noise.exp()
        res += -0.5 * self.log_noise - 0.5 * math.log(2 * math.pi)
        return res.sum(0)
