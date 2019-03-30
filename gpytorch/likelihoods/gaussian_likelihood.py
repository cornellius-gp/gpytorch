#!/usr/bin/env python3

import math

from torch.nn.functional import softplus

from .. import settings
from ..likelihoods import Likelihood
from ..distributions import base_distributions
from .noise_models import HomoskedasticNoise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models. """

    def __init__(self, noise_covar):
        super().__init__()
        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape, *params):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape
        return self.noise_covar(*params, shape=shape)

    def forward(self, function_samples, *params, **kwargs):
        return base_distributions.Normal(
            function_samples, self._shaped_noise_covar(function_samples.shape, *params).diag()
        )

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        full_covar = covar + self._shaped_noise_covar(mean.shape, *params)
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, **kwargs):
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            batch_size=batch_size,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
        )
        super().__init__(noise_covar=noise_covar)

    def _param_transform(self, value):
        return self.noise_covar._param_transform(value)

    def _inv_param_transform(self, value):
        return self.noise_covar._inv_param_transform(value)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self):
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value):
        self.noise_covar.initialize(raw_noise=value)

    def expected_log_prob(self, target, input, *params, **kwargs):
        mean, variance = input.mean, input.variance
        noise = self.noise_covar.noise

        if mean.dim() > target.dim():
            target = target.unsqueeze(-1)

        if variance.ndimension() == 1:
            if settings.debug.on() and noise.size(0) > 1:
                raise RuntimeError("With batch_size > 1, expected a batched MultivariateNormal distribution.")
            noise = noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / noise
        res += -0.5 * noise.log() - 0.5 * math.log(2 * math.pi)
        return res.sum(-1)
