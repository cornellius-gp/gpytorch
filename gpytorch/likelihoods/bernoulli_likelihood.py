#!/usr/bin/env python3

import warnings
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Bernoulli

from ..distributions import base_distributions, MultivariateNormal
from ..functions import log_normal_cdf
from .likelihood import _OneDimensionalLikelihood


class BernoulliLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF :math:`\Phi(x)`). Given the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood compactly as:

    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi((2y - 1)f)
        \end{equation*}

    .. note::
        BernoulliLikelihood has an analytic marginal distribution.

    .. note::
        The labels should take values in {0, 1}.
    """

    has_analytic_marginal: bool = True

    def __init__(self) -> None:
        return super().__init__()

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Bernoulli:
        output_probs = base_distributions.Normal(0, 1).cdf(function_samples)
        return base_distributions.Bernoulli(probs=output_probs)

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> Bernoulli:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = base_distributions.Normal(0, 1).cdf(link)
        return base_distributions.Bernoulli(probs=output_probs)

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        if torch.any(observations.eq(-1)):
            # Remove after 1.0
            warnings.warn(
                "BernoulliLikelihood.expected_log_prob expects observations with labels in {0, 1}. "
                "Observations with labels in {-1, 1} are deprecated.",
                DeprecationWarning,
            )
        else:
            observations = observations.mul(2).sub(1)
        # Custom function here so we can use log_normal_cdf rather than Normal.cdf
        # This is going to be less prone to overflow errors
        log_prob_lambda = lambda function_samples: log_normal_cdf(function_samples.mul(observations))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob
