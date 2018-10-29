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

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        noise = self.noise
        lazy_covar_f = variational_dist_f.lazy_covariance_matrix
        if lazy_covar_f.ndimension() == 2:
            noise = noise.squeeze(0)
        y_lazy_covar = add_diag(lazy_covar_f, noise)
        y_mean = variational_dist_f.mean
        y_dist = MultivariateNormal(y_mean, y_lazy_covar)
        if len(y_dist.shape()) > 1:
            pyro.sample(name_prefix + "._training_labels", y_dist.independent(1), obs=y_obs)
        else:
            pyro.sample(name_prefix + "._training_labels", y_dist, obs=y_obs)
