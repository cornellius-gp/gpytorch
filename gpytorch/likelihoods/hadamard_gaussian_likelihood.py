#!/usr/bin/env python3

from typing import Any

import torch

from linear_operator.operators import DiagLinearOperator

from ..distributions import base_distributions, MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .noise_models import MultitaskHomoskedasticNoise


class HadamardGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Likelihood for input-wise homoskedastic and task-wise heteroskedastic noise,
    i.e. we learn a different (constant) noise level for each fidelity.

    Args:
        num_of_tasks: Number of tasks in the multi-output GP.
        noise_prior: Prior for the noise.
        noise_constraint: Constraint on the noise value.
    """

    def __init__(
        self,
        num_tasks,
        noise_prior=None,
        noise_constraint=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        self.num_tasks = num_tasks
        super().__init__(noise_covar=noise_covar)

    @property
    def noise(self) -> torch.Tensor:
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self) -> torch.Tensor:
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value: torch.Tensor) -> None:
        self.noise_covar.initialize(raw_noise=value)

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any):
        # params contains task indexes
        task_idcs = params[0][-1]
        noise_base_covar_matrix = self.noise_covar(*params, shape=base_shape, **kwargs)

        all_tasks = torch.arange(self.num_tasks)[:, None]
        diag = torch.eq(all_tasks, task_idxs.mT)
        mask = DiagLinearOperator(diag)
        return (noise_base_covar_matrix @ mask).sum(dim=-3)

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> base_distributions.Normal:
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(function_samples, noise.sqrt())

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs).squeeze(0)
        full_covar = covar + noise_covar
        return function_dist.__class__(mean, full_covar)
