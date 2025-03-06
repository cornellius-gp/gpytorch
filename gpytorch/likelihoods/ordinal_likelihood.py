from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.distributions import Categorical

from ..constraints import Interval, Positive
from ..distributions import MultivariateNormal
from ..priors import Prior
from .likelihood import _OneDimensionalLikelihood


def inv_probit(x, jitter=1e-3):
    """
    Inverse probit function (standard normal CDF) with jitter for numerical stability.

    Args:
        x: Input tensor
        jitter: Small constant to ensure outputs are strictly between 0 and 1

    Returns:
        Probabilities between jitter and 1-jitter
    """
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) * (1 - 2 * jitter) + jitter


class OrdinalLikelihood(_OneDimensionalLikelihood):
    r"""
    An ordinal likelihood for regressing over ordinal data.

    The data are integer values from :math:`0` to :math:`k`, and the user must specify :math:`(k-1)`
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be :math:`[a_0, a_1, ... a_{k-1}]`, then the likelihood is

    .. math::
        p(Y=0|F) &= \Phi((a_0 - F) / \sigma)

        p(Y=1|F) &= \Phi((a_1 - F) / \sigma) - \Phi((a_0 - F) / \sigma)

        p(Y=2|F) &= \Phi((a_2 - F) / \sigma) - \Phi((a_1 - F) / \sigma)

        ...

        p(Y=K|F) &= 1 - \Phi((a_{k-1} - F) / \sigma)

    where :math:`\Phi` is the cumulative density function of a Gaussian (the inverse probit
    function) and :math:`\sigma` is a parameter to be learned.

    From Chu et Ghahramani, Journal of Machine Learning Research, 2005
    [https://www.jmlr.org/papers/volume6/chu05a/chu05a.pdf].

    :param bin_edges: A tensor of shape :math:`(k-1)` containing the bin edges.
    :param batch_shape: The batch shape of the learned sigma parameter (default: []).
    :param sigma_prior: Prior for sigma parameter :math:`\sigma`.
    :param sigma_constraint: Constraint for sigma parameter :math:`\sigma`.

    :ivar torch.Tensor bin_edges: :math:`\{a_i\}_{i=0}^{k-1}` bin edges
    :ivar torch.Tensor sigma: :math:`\sigma` parameter (scale)
    """

    def __init__(
        self,
        bin_edges: Tensor,
        batch_shape: torch.Size = torch.Size([]),
        sigma_prior: Optional[Prior] = None,
        sigma_constraint: Optional[Interval] = None,
    ) -> None:
        super().__init__()

        self.num_bins = len(bin_edges) + 1
        self.register_parameter("bin_edges", torch.nn.Parameter(bin_edges, requires_grad=False))

        if sigma_constraint is None:
            sigma_constraint = Positive()

        self.raw_sigma = torch.nn.Parameter(torch.ones(*batch_shape, 1))
        if sigma_prior is not None:
            self.register_prior("sigma_prior", sigma_prior, lambda m: m.sigma, lambda m, v: m._set_sigma(v))

        self.register_constraint("raw_sigma", sigma_constraint)

    @property
    def sigma(self) -> Tensor:
        return self.raw_sigma_constraint.transform(self.raw_sigma)

    @sigma.setter
    def sigma(self, value: Tensor) -> None:
        self._set_sigma(value)

    def _set_sigma(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma)
        self.initialize(raw_sigma=self.raw_sigma_constraint.inverse_transform(value))

    def forward(self, function_samples: Tensor, *args: Any, data: Dict[str, Tensor] = {}, **kwargs: Any) -> Categorical:
        if isinstance(function_samples, MultivariateNormal):
            function_samples = function_samples.sample()

        # Compute scaled bin edges
        scaled_edges = self.bin_edges / self.sigma
        scaled_edges_left = torch.cat([scaled_edges, torch.tensor([torch.inf], device=scaled_edges.device)], dim=-1)
        scaled_edges_right = torch.cat([torch.tensor([-torch.inf], device=scaled_edges.device), scaled_edges])

        # Calculate cumulative probabilities using standard normal CDF (probit function)
        function_samples = function_samples.unsqueeze(-1)
        scaled_edges_left = scaled_edges_left.reshape(1, -1)
        scaled_edges_right = scaled_edges_right.reshape(1, -1)
        probs = inv_probit(scaled_edges_left - function_samples / self.sigma) - inv_probit(
            scaled_edges_right - function_samples / self.sigma
        )

        return Categorical(probs=probs)
