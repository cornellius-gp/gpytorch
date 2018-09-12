from __future__ import absolute_import, division, print_function, unicode_literals

import math

import torch

from ..distributions import MultivariateNormal
from ..functions import add_diag
from ..likelihoods import Likelihood
from ..priors._compatibility import _bounds_to_prior


class GaussianLikelihood(Likelihood):
    def __init__(self, log_noise_prior=None, log_noise_bounds=None):
        # TODO: Remove deprecated log_noise_bounds kwarg
        log_noise_prior = _bounds_to_prior(prior=log_noise_prior, bounds=log_noise_bounds)
        super(GaussianLikelihood, self).__init__()
        self.register_parameter(name="log_noise", parameter=torch.nn.Parameter(torch.zeros(1)), prior=log_noise_prior)

    def forward(self, input):
        assert isinstance(input, MultivariateNormal)
        mean, covar = input.representation()
        noise = add_diag(covar, self.log_noise.exp())
        return input.__class__(mean, noise)

    def log_probability(self, input, target):
        res = -0.5 * ((target - input.mean()) ** 2 + input.var()) / self.log_noise.exp()
        res += -0.5 * self.log_noise - 0.5 * math.log(2 * math.pi)
        return res.sum(0)
