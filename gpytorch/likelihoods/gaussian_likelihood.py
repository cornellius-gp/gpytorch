from __future__ import absolute_import, division, print_function, unicode_literals

import math

from .. import settings
from ..distributions import MultivariateNormal
from ..functions import add_diag
from .likelihood import Likelihood
from .noise_models import HomoskedasticNoise


class _GaussianLikelihoodBase(Likelihood):
    def __init__(self, noise_covar):
        super(_GaussianLikelihoodBase, self).__init__()
        self.noise_covar = noise_covar

    def forward(self, input, *params):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("Gaussian likelihoods require a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        full_covar = covar + self.noise_covar(*params)
        return input.__class__(mean, full_covar)

    def variational_log_probability(self, input, target):
        raise NotImplementedError


class GaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, log_noise_prior=None, batch_size=1):
        noise_covar = HomoskedasticNoise(log_noise_prior=log_noise_prior, batch_size=1)
        super(GaussianLikelihood, self).__init__(noise_covar=noise_covar)

    def variational_log_probability(self, input, target):
        mean, variance = input.mean, input.variance
        if mean.dim() > target.dim():
            target = target.unsqueeze(-1)

        log_noise = self.log_noise_covar.log_noise
        if variance.ndimension() == 1:
            if settings.debug.on() and log_noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            log_noise = log_noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / log_noise.exp()
        res += -0.5 * log_noise - 0.5 * math.log(2 * math.pi)

        if res.dim() == 1:
            return res.sum()
        else:
            return res.sum(-2).squeeze(-1)

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
