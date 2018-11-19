#!/usr/bin/env python3

import math
import torch
from ..distributions import MultivariateNormal
from ..functions import add_diag
from ..likelihoods import Likelihood
from ..lazy import DiagLazyTensor
from .. import settings
from ..utils.deprecation import _deprecate_kwarg
from ..utils.transforms import _get_inv_param_transform
from torch.nn.functional import softplus


class GaussianLikelihood(Likelihood):
    r"""
    """

    def __init__(self, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, **kwargs):
        noise_prior = _deprecate_kwarg(kwargs, "log_noise_prior", "noise_prior", noise_prior)
        super(GaussianLikelihood, self).__init__()
        self._param_transform = param_transform
        self._inv_param_transform = _get_inv_param_transform(param_transform, inv_param_transform)
        self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(batch_size, 1)))
        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda: self.noise, lambda v: self._set_noise(v))

    @property
    def noise(self):
        return self._param_transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        self.initialize(raw_noise=self._inv_param_transform(value))

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

        res = -0.5 * ((target - mean) ** 2 + variance) / self.noise
        res += -0.5 * log_noise - 0.5 * math.log(2 * math.pi)
        return res.sum(-1)

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        noise = self.noise
        var_f = variational_dist_f.lazy_covariance_matrix.diag()
        y_mean = variational_dist_f.mean
        if y_mean.dim() == 1:
            noise = noise.squeeze(0)
        y_lazy_covar = DiagLazyTensor(var_f + noise.expand_as(var_f))
        y_dist = MultivariateNormal(y_mean, y_lazy_covar)
        pyro.sample(name_prefix + "._training_labels", y_dist, obs=y_obs)
