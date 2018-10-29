from __future__ import absolute_import, division, print_function, unicode_literals

import math
import warnings
import logging

from .. import settings
from ..distributions import MultivariateNormal
from ..lazy import AddedDiagLazyTensor, DiagLazyTensor
from .likelihood import Likelihood
from .noise_models import HomoskedasticNoise

DEPRECATION_WARNING = "'GaussianLikelihood' was renamed to 'HomoskedasticGaussianLikelihood'"


class GaussianLikelihood(Likelihood):
    def __init__(self, *args, **kwargs):
        if len(args) + len(kwargs) == 0 or "log_noise_prior" in kwargs or "batch_size" in kwargs:
            warnings.warn(DEPRECATION_WARNING, DeprecationWarning)
            logging.warning(DEPRECATION_WARNING)
            self.__init__(log_noise_covar=HomoskedasticNoise(*args, **kwargs))
            self._is_homoskedastic = True
        else:
            super(GaussianLikelihood, self).__init__()
            self.log_noise_covar = args[0] if len(kwargs) == 0 else kwargs["log_noise_covar"]

    def forward(self, input, *params):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("GaussianLikelihood requires a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        log_noise_covar = self.log_noise_covar(*params)
        if isinstance(log_noise_covar, DiagLazyTensor):
            full_covar = AddedDiagLazyTensor(covar, log_noise_covar.exp())
        else:
            # TODO: Deal with non-diagonal noise covariance models
            full_covar = covar + log_noise_covar.exp()
        return input.__class__(mean, full_covar)

    def variational_log_probability(self, input, target):
        if hasattr(self, "_is_homoskedastic"):
            return HomoskedasticGaussianLikelihood.variational_log_probability(self, input, target)
        else:
            raise NotImplementedError


class HomoskedasticGaussianLikelihood(GaussianLikelihood):
    def __init__(self, log_noise_prior=None, batch_size=1):
        log_noise_covar = HomoskedasticNoise(log_noise_prior=log_noise_prior, batch_size=1)
        super(HomoskedasticGaussianLikelihood, self).__init__(log_noise_covar=log_noise_covar)

    def variational_log_probability(self, input, target):
        mean, variance = input.mean, input.variance
        log_noise = self.log_noise_covar.log_noise
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
