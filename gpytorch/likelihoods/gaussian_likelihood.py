#!/usr/bin/env python3

import math

from torch.nn.functional import softplus

from .. import settings
from ..distributions import MultivariateNormal
from ..likelihoods import Likelihood
from ..lazy import BlockDiagLazyTensor, DiagLazyTensor
from ..utils.deprecation import _deprecate_kwarg
from .noise_models import HomoskedasticNoise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models. """

    def __init__(self, noise_covar):
        super().__init__()
        self.noise_covar = noise_covar

    def forward(self, input, *params):
        if not isinstance(input, MultivariateNormal):
            raise ValueError("Gaussian likelihoods require a MultivariateNormal input")
        mean, covar = input.mean, input.lazy_covariance_matrix
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = mean.shape
        noise_covar = self.noise_covar(*params, shape=shape)
        full_covar = covar + noise_covar
        return input.__class__(mean, full_covar)

    def variational_log_probability(self, input, target):
        raise NotImplementedError


class GaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, **kwargs):
        noise_prior = _deprecate_kwarg(kwargs, "log_noise_prior", "noise_prior", noise_prior)
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
        self.noise_covar.initialize(value)

    @property
    def raw_noise(self):
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value):
        self.noise_covar.initialize(raw_noise=value)

    def variational_log_probability(self, input, target):
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

    def pyro_sample_y(self, variational_dist_f, y_obs, sample_shape, name_prefix=""):
        import pyro

        noise = self.noise
        var_f = variational_dist_f.lazy_covariance_matrix.diag()
        y_mean = variational_dist_f.mean
        if y_mean.dim() == 1:
            noise = noise.squeeze(0)
        y_lazy_covar = DiagLazyTensor(var_f + noise.expand_as(var_f))
        y_dist = MultivariateNormal(y_mean, y_lazy_covar)
        if len(y_dist.batch_shape):
            y_dist = y_dist.__class__(
                y_dist.mean.contiguous().view(-1), BlockDiagLazyTensor(y_dist.lazy_covariance_matrix)
            )
            y_obs = y_obs.view_as(y_dist.mean)
        pyro.sample(name_prefix + "._training_labels", y_dist, obs=y_obs)
