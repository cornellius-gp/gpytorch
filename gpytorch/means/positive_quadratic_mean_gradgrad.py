#!/usr/bin/env python3

import torch

from .mean import Mean


class PositiveQuadraticMeanGradGrad(Mean):
    r"""
    A positive quadratic prior mean function and its first and second derivative, i.e.:

    .. math::

        \mu(\mathbf x) &= \frac12 \mathbf x^\top \cdot A \cdot \mathbf x \\
        \nabla \mu(\mathbf x) &= \mathbf x \cdot A \\
        \nabla^2 \mu(\mathbf x) &= \mathbf A \\

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
        dres = torch.zeros(*x.shape, device=x.device)
        ddres = torch.zeros(*x.shape, device=x.device)
        for i in range(x.shape[-2]):
            for j in range(self.dim):
                for k in range(j + 1):
                    c = self.cholesky[..., j * (j + 1) // 2 + k]
                    dres[..., i, j] += self.cholesky[..., j * (j + 1) // 2 + k] * xl[..., i, k]
                    ddres[..., i, j] += c**2
        return torch.cat((res.unsqueeze(-1), dres, ddres), -1)
