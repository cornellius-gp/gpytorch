from __future__ import absolute_import, division, print_function, unicode_literals

import math
import torch
from ..distributions import MultivariateNormal
from ..functions import add_diag
from ..likelihoods import Likelihood
from .. import settings


class GaussianLikelihood(Likelihood):
    r"""
    """

    def __init__(self, log_noise_prior=None, batch_size=1):
        super(GaussianLikelihood, self).__init__()
        self.register_parameter(
            name="log_noise", parameter=torch.nn.Parameter(torch.zeros(batch_size, 1)), prior=log_noise_prior
        )

    @property
    def noise(self):
        return self.log_noise.exp()

    def forward(self, input):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("GaussianLikelihood requires a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        noise = self.noise
        if covar.ndimension() == 2:
            if settings.debug.on() and noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            noise = noise.squeeze(0)

        return input.__class__(mean, add_diag(covar, noise))

    def variational_log_probability(self, input, target):
        mean, variance = input.mean, input.variance
        log_noise = self.log_noise
        if variance.ndimension() == 1:
            if settings.debug.on() and log_noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            log_noise = log_noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / log_noise.exp()
        res += -0.5 * log_noise - 0.5 * math.log(2 * math.pi)
        return res.sum(0)
