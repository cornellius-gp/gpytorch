#!/usr/bin/env python3

import torch

from .mean import Mean


class PositiveQuadraticMean(Mean):
    r"""
    A positive quadratic prior mean function and its first derivative, i.e.:

    .. math::

        \mu(\mathbf x) &= \frac12 \mathbf x^\top \cdot A \cdot \mathbf x

    where :math:`A = L L^\top` and L a lower triangular matrix.

    :param input_size: dimension of input :math:`\mathbf x`.
    :type input_size: int
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor cholesky: vector containing :math:`L` components.
    """

    def __init__(self, input_size: int, batch_shape: torch.Size = torch.Size()):
        super().__init__()
        self.dim = input_size
        self.register_parameter(
            name="cholesky", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size * (input_size + 1) // 2))
        )

    def forward(self, x):
        xl = torch.zeros(*x.shape, device=x.device)
        for i in range(x.shape[-2]):
            for j in range(self.dim):
                for k in range(j, self.dim):
                    xl[..., i, j] += self.cholesky[..., k * (k + 1) // 2 + j] * x[..., i, k]
        res = xl.pow(2).sum(-1).div(2)
        return res
