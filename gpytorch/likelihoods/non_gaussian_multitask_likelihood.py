#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch.distributions import Distribution as _Distribution

from ..distributions import MultivariateNormal
from .likelihood import Likelihood


class NonGaussianMultitaskLikelihood(Likelihood):
    r"""
    Wraps a one-dimensional likelihood (e.g. :obj:`~gpytorch.likelihoods.BernoulliLikelihood`) so that it can be
    used with multitask models (e.g. :obj:`~gpytorch.variational.LMCVariationalStrategy`) whose function
    distribution is a :obj:`~gpytorch.distributions.MultitaskMultivariateNormal` with a (... x N x T) mean.

    The ``base_likelihood`` is applied independently to every task and data point, and the terms are summed
    over the :math:`T` tasks, so that :meth:`expected_log_prob` and :meth:`log_marginal` return a (... x N) tensor,
    as :obj:`~gpytorch.likelihoods.MultitaskGaussianLikelihood` does.

    :param base_likelihood: The one-dimensional likelihood that is applied to every task.

    Example:
        >>> likelihood = gpytorch.likelihoods.NonGaussianMultitaskLikelihood(gpytorch.likelihoods.BernoulliLikelihood())
        >>> mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(-2))
        >>> loss = -mll(model(train_x), train_y)  # train_y is (N x T), with labels in {0, 1}
    """

    def __init__(self, base_likelihood: Likelihood) -> None:
        super().__init__()
        self.base_likelihood = base_likelihood

    @property
    def has_analytic_marginal(self) -> bool:
        return getattr(self.base_likelihood, "has_analytic_marginal", False)

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> _Distribution:
        return self.base_likelihood.forward(function_samples, *args, **kwargs)

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        return self.base_likelihood.expected_log_prob(observations, function_dist, *args, **kwargs).sum(-1)

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        return self.base_likelihood.log_marginal(observations, function_dist, *args, **kwargs).sum(-1)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> _Distribution:
        return self.base_likelihood.marginal(function_dist, *args, **kwargs)
