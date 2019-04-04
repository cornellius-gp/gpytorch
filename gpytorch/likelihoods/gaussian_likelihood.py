#!/usr/bin/env python3

import math
import warnings
from typing import Any

import torch
from torch import Tensor

from .. import settings
from ..distributions import MultivariateNormal, base_distributions
from ..likelihoods import Likelihood
from .noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    def __init__(self, noise_covar: Noise, **kwargs: Any) -> None:

        super().__init__()
        param_transform = kwargs.get("param_transform")
        if param_transform is not None:
            warnings.warn("The 'param_transform' argument is now deprecated. If you want to use a different "
                          "transformaton, specify a different 'lengthscale_constraint' instead.")

        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape
        return self.noise_covar(*params, shape=shape, **kwargs)

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> base_distributions.Normal:
        observation_noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(function_samples, observation_noise)

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_size=1, **kwargs):
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_size=batch_size,
        )
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def expected_log_prob(self, target: Tensor, input: MultivariateNormal, *params: Any, **kwargs: Any) -> Tensor:
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


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise: Tensor, **kwargs: Any) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)
