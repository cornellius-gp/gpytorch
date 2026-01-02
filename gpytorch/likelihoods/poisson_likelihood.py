from typing import Any

import torch
from torch import Tensor
from torch.distributions import Poisson

from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class PoissonLikelihood(_OneDimensionalLikelihood):
    r"""
    A Poisson likelihood for regressing over count data.

    The Poisson distribution is parameterized by :math:`\lambda > 0` (rate parameter),
    which represents the expected number of events occurring in a fixed interval.

    The rate parameter is derived from the GP function samples through a softplus transformation:

    .. math::
        \begin{equation*}
            \lambda = \text{softplus}(f)
        \end{equation*}

    where :math:`f` is the GP function sample.

    The likelihood is then:

    .. math::
        p(y \mid f) = \text{Poisson}(\lambda)

    This likelihood does not have learnable parameters and
    enforces nonnegativity through the softplus activation function.

    :ivar torch.Tensor rate: :math:`\lambda` parameter (rate)
    """

    def __init__(self):
        super().__init__()

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Poisson:
        rates = torch.nn.functional.softplus(function_samples)
        return base_distributions.Poisson(rates)
