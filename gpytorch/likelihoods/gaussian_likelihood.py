from __future__ import absolute_import, division, print_function, unicode_literals

import math

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor
from .homoskedastic_noise import HomoskedasticNoise
from .likelihood import Likelihood


class GaussianLikelihood(Likelihood):
    def __init__(self, noise_covar):
        Likelihood.__init__(self)
        self.noise_covar = noise_covar

    def forward(self, input, *params):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("GaussianLikelihood requires a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        return input.__class__(mean, covar + self.noise_covar(*params))


class HomoskedasticGaussianLikelihood(GaussianLikelihood):
    def __init__(self, log_noise_prior=None, batch_size=1):
        noise_covar = HomoskedasticNoise(log_noise_prior=log_noise_prior, batch_size=1)
        super(HomoskedasticGaussianLikelihood, self).__init__(noise_covar)

    def variational_log_probability(self, input, target):
        mean, variance = input.mean, input.variance
        log_noise = self.noise_covar.log_noise
        if variance.ndimension() == 1:
            if settings.debug.on() and log_noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            log_noise = log_noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / log_noise.exp()
        res += -0.5 * log_noise - 0.5 * math.log(2 * math.pi)
        return res.sum(0)


class HeteroskedasticGaussianLikelihood(GaussianLikelihood):
    def __init__(self, log_noise_model):
        Likelihood.__init__(self)
        self.log_noise_model = log_noise_model

    def forward(self, input, *params):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("HeteroskedasticGaussianLikelihood requires a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        # TODO: This is inefficient, fix it! Allow for non-diagonal outputs
        log_noise_covar = self.log_noise_model(*params).diag()
        noise_covar = DiagLazyTensor(log_noise_covar.exp())
        return input.__class__(mean, covar + noise_covar)
