#!/usr/bin/env python3

from typing import Any

import torch

from ..distributions import base_distributions, MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .noise_models import MultitaskHomoskedasticNoise


class HadamardGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Likelihood for input-wise homo-skedastic noise, and task-wise
    hetero-skedastic, i.e. we learn a different (constant) noise level for each fidelity.

    Args:
        num_of_tasks : number of tasks in the multi output GP
        noise_prior : any prior you want to put on the noise
        noise_constraint : constraint to put on the noise
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
        # params contains training data
        task_idxs = params[0][-1]
        noise_base_covar_matrix = self.noise_covar(*params, shape=base_shape, **kwargs)
        # initialize masking
        mask = torch.zeros(size=noise_base_covar_matrix.shape)
        # for each task create a masking
        for task_num in range(self.num_tasks):
            # create vector of indexes
            task_idx_diag = (task_idxs == task_num).int().reshape(-1).diag()
            mask[..., task_num, :, :] = task_idx_diag
        # multiply covar by masking
        # there seems to be problems when base_shape is singleton, so we need to squeeze
        if base_shape == torch.Size([1]):
            noise_base_covar_matrix = noise_base_covar_matrix.squeeze(-1).mul(mask.squeeze(-1))
            noise_covar_matrix = noise_base_covar_matrix.unsqueeze(-1).sum(dim=1)
        else:
            noise_covar_matrix = noise_base_covar_matrix.mul(mask).sum(dim=1)
        return noise_covar_matrix

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
