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

    The Negative Binomial distribution is parameterized by :math:`r > 0` (total count)
    and :math:`p \in (0, 1)` (probability of success).
    The total count parameter is derived from the GP function samples.

    We parameterize the likelihood as follows:

    .. math::
        \begin{equation*}
            r = \text{softplus}(f) \cdot \frac{1-p}{p}, \quad p = \sigma(\text{raw\_probs})
        \end{equation*}

    where :math:`f` is the GP function sample and :math:`\sigma(\cdot)` is the sigmoid function.
    Therefore, the GP function parametrizes the mean of the negative binomial distribution.
    When :math:`\text{total\_count\_param}` is True, GP directly parametrizes :math:`r = \text{softplus}(f)`.

    The likelihood is then:

    .. math::
        p(y \mid f) = \text{NegativeBinomial} \left( r, p \right).

    :param batch_shape: The batch shape of the learned probabilities parameter (default: []).
    :param probs_prior: Prior for probabilities parameter :math:`p`.
    :param probs_constraint: Constraint for probabilities parameter :math:`p`.
    :param total_count_param: Whether the GP parametrizes the total count parameter :math:`r` (default: False).

    :ivar torch.Tensor probs: :math:`p` parameter (probability of success)

    **Reference:**
        S. Damato et al., Forecasting intermittent time series with Gaussian Processes and Tweedie likelihood.
        International Journal of Forecasting, 2025.
    """

    has_analytic_marginal: bool = True

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        probs_prior: Optional[Prior] = None,
        probs_constraint: Optional[Interval] = None,
        total_count_param: bool = False,
    ) -> None:
        super().__init__()

        if probs_constraint is None:
            probs_constraint = Interval(0, 1)

        self.raw_probs = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        if probs_prior is not None:
            self.register_prior("probs_prior", probs_prior, lambda m: m.probs, lambda m, v: m._set_probs(v))

        self.register_constraint("raw_probs", probs_constraint)

        self.total_count_param = total_count_param

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
        if self.total_count_param:
            total_count = torch.nn.functional.softplus(function_samples)
        else:
            total_count = torch.nn.functional.softplus(function_samples) * (1 - probs) / probs
        return base_distributions.NegativeBinomial(total_count=total_count, probs=probs)
