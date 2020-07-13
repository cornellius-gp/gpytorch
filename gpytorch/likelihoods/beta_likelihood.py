#!/usr/bin/env python3

import torch

from ..distributions import base_distributions
from .likelihood import _OneDimensionalLikelihood


class BetaLikelihood(_OneDimensionalLikelihood):
    r"""
    A Beta likelihood for regressing over percentages.

    The Beta distribution is parameterized by :math:`\alpha > 0` and :math:`\beta > 0` parameters
    which roughly correspond to the number of prior positive and negative observations.
    We instead parameterize it through a mixture :math:`m \in [0, 1]` and scale :math:`s > 0` parameter.

    .. math::
        \begin{equation*}
            \alpha = ms, \quad \beta = (1-m)s
        \end{equation*}

    The mixture parameter is the output of the GP passed through a logit function.
    The scale parameter is learned.

    :var torch.Tensor scale: :math:`s` parameter (scale)
    """

    def __init__(self, batch_shape=torch.Size([])):
        super().__init__()
        self.raw_scale = torch.nn.Parameter(torch.ones(*batch_shape, 1))

    @property
    def scale(self):
        return torch.exp(self.raw_scale)

    def forward(self, function_samples, **kwargs):
        mixture = torch.sigmoid(function_samples)
        scale = self.scale
        alpha = mixture * scale + 1
        beta = scale - alpha + 2
        return base_distributions.Beta(concentration1=alpha, concentration0=beta)
