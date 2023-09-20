#!/usr/bin/env python3

from typing import Any, Optional

import torch

from ..priors import Prior
from .mean import Mean


class ConstantMeanGradGrad(Mean):
    r"""
    A (non-zero) constant prior mean function and its first and second derivatives, i.e.:

    .. math::

        \mu(\mathbf x) &= C \\
        \nabla \mu(\mathbf x) &= \mathbf 0 \\
        \nabla^2 \mu(\mathbf x) &= \mathbf 0

    where :math:`C` is a learned constant.

    :param prior: Prior for constant parameter :math:`C`.
    :type prior: ~gpytorch.priors.Prior, optional
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor constant: :math:`C` parameter
    """

    def __init__(
        self,
        prior: Optional[Prior] = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ):
        super(ConstantMeanGradGrad, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        batch_shape = torch.broadcast_shapes(self.batch_shape, input.shape[:-2])
        mean = self.constant.unsqueeze(-1).expand(*batch_shape, input.size(-2), 2 * input.size(-1) + 1).contiguous()
        mean[..., 1:] = 0
        return mean
