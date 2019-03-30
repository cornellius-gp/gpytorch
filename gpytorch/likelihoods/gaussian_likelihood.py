#!/usr/bin/env python3

import math
from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn.functional import softplus

from .. import settings
from ..distributions import MultivariateNormal, base_distributions
from ..lazy import DiagLazyTensor
from ..likelihoods import Likelihood
from .noise_models import FixedGaussianNoise, HomoskedasticNoise, Noise


class _GaussianLikelihoodBase(Likelihood):
    """Base class for Gaussian Likelihoods, supporting general heteroskedastic noise models."""

    def __init__(self, noise_covar: Noise) -> None:
        super().__init__()
        self.noise_covar = noise_covar

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any):
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape
        return self.noise_covar(*params, shape=shape)

    def forward(
        self, function_samples: Tensor, *params: Any, observation_noise: Optional[Tensor] = None, **kwargs: Any
    ):
        if observation_noise is None:
            self._shaped_noise_covar(function_samples.shape, *params).diag()
        else:
            var = observation_noise
        return base_distributions.Normal(function_samples, var)

    def marginal(
        self, function_dist: MultivariateNormal, *params: Any, observation_noise: Optional[Tensor] = None, **kwargs: Any
    ) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        if observation_noise is None:
            noise_covar = self._shaped_noise_covar(mean.shape, *params)
        else:
            noise_covar = DiagLazyTensor(observation_noise)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)


class GaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(
        self,
        noise_prior=None,
        batch_size=1,
        param_transform=softplus,
        inv_param_transform=None,
        **kwargs
    ):
        noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior,
            batch_size=batch_size,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
        )
        super().__init__(noise_covar=noise_covar)

    def _param_transform(self, value: Tensor) -> Tensor:
        return self.noise_covar._param_transform(value)

    def _inv_param_transform(self, value: Tensor) -> Tensor:
        return self.noise_covar._inv_param_transform(value)

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
                raise RuntimeError(
                    "With batch_size > 1, expected a batched MultivariateNormal distribution."
                )
            noise = noise.squeeze(0)

        res = -0.5 * ((target - mean) ** 2 + variance) / noise
        res += -0.5 * noise.log() - 0.5 * math.log(2 * math.pi)
        return res.sum(-1)


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise: Tensor, **kwargs) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self.noise_covar.initialize(noise=value)
