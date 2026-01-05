from typing import Any, Optional

import torch
from torch import Tensor
from torch.distributions import NegativeBinomial

from ..constraints import Interval
from ..distributions import base_distributions
from ..priors import Prior
from .likelihood import _OneDimensionalLikelihood


class NegativeBinomialLikelihood(_OneDimensionalLikelihood):
    r"""
    A Negative Binomial likelihood for regressing over count data.

    This likelihood is parameterized by :math:`k > 0`, the total number of failures (also named total count
    in `torch.distributions`), and :math:`p \in (0, 1)`, the probability of success.

    Under this parameterization, the random variable represents the number of successful independent trials,
    each with probability of success :math:`p`, observed before :math:`k` failures occur. The likelihood is:

    .. math:: p(y \mid f) = \text{NegativeBinomial} \left( k, p \right).

    The number of failures parameter is derived as:

    .. math::
        \begin{equation*}
            k = \text{softplus}(f) \cdot \frac{1 - p}{p}
        \end{equation*}

    where :math:`f` is the GP function sample. With this choice,
    the GP function parametrizes the mean of the negative binomial distribution.
    When :attr:`num_failures_param` is True, the GP directly parametrizes :math:`k = \text{softplus}(f)`.

    :param batch_shape: The batch shape of the learned probabilities parameter (default: []).
    :param probs_prior: Prior for probabilities parameter :math:`p`.
    :param probs_constraint: Constraint for probabilities parameter :math:`p`.
    :param num_failures_param: Whether the GP parametrizes the number of failures parameter :math:`k` (default: False).

    :ivar torch.Tensor probs: :math:`p` parameter (probability of success)

    **Reference:**
        Damato et al. (2025), Forecasting intermittent time series with Gaussian Processes and Tweedie likelihood.
    """

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        probs_prior: Optional[Prior] = None,
        probs_constraint: Optional[Interval] = None,
        num_failures_param: bool = False,
    ) -> None:
        super().__init__()

        if probs_constraint is None:
            probs_constraint = Interval(0, 1)

        self.raw_probs = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if probs_prior is not None:
            self.register_prior("probs_prior", probs_prior, lambda m: m.probs, lambda m, v: m._set_probs(v))

        self.register_constraint("raw_probs", probs_constraint)

        self.num_failures_param = num_failures_param

    @property
    def probs(self) -> Tensor:
        return self.raw_probs_constraint.transform(self.raw_probs)

    @probs.setter
    def probs(self, value: Tensor) -> None:
        self._set_probs(value)

    def _set_probs(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_probs)
        self.initialize(raw_probs=self.raw_probs_constraint.inverse_transform(value))

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> NegativeBinomial:
        probs = torch.clamp(self.probs, 1e-06, 1 - 1e-06)
        if self.num_failures_param:
            num_failures = torch.nn.functional.softplus(function_samples)
        else:
            num_failures = torch.nn.functional.softplus(function_samples) * (1 - probs) / probs
        return base_distributions.NegativeBinomial(total_count=num_failures, probs=probs)
