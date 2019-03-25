#!/usr/bin/env python3

from torch import Tensor

from ..distributions import base_distributions
from .gaussian_likelihood import _GaussianLikelihoodBase
from .noise_models import FixedGaussianNoise


class FixedNoiseGaussianLikelihood(_GaussianLikelihoodBase):
    def __init__(self, noise: Tensor, **kwargs) -> None:
        super().__init__(noise_covar=FixedGaussianNoise(noise=noise))

    @property
    def noise(self) -> Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value) -> None:
        self.noise_covar.initialize(noise=value)

    def forward(self, function_samples, *params, **kwargs):
        return base_distributions.Normal(
            function_samples, self._shaped_noise_covar(function_samples.shape, *params)
        )

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        if not self.noise.shape[-1] == covar.shape[-1]:
            raise RuntimeError(
                "Shape of fixed noise ({}) incompatible with that of covariance matrix ({})".format(
                    self.noise.shape, covar.shape
                )
            )
        full_covar = covar + self._shaped_noise_covar(mean.shape, *params)
        return function_dist.__class__(mean, full_covar)
