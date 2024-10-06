#!/usr/bin/env python3

import torch

from .mean import Mean


class QuadraticMeanGradGrad(Mean):
    r"""
    A quadratic prior mean function and its first and second derivative, i.e.:

    .. math::

        \mu(\mathbf x) &= \frac12 \mathbf x^\top \cdot A \cdot \mathbf x \\
        \nabla \mu(\mathbf x) &= \frac12 \mathbf x \cdot ( A + A^\top ) \\
        \nabla^2 \mu(\mathbf x) &= \mathbf A \\

    where :math:`A` and L a square matrix.

    :param input_size: dimension of input :math:`\mathbf x`.
    :type input_size: int
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor A: a square matrix.
    """

    def __init__(self, input_size: int, batch_shape: torch.Size = torch.Size()):
        super().__init__()
        self.dim = input_size
        self.register_parameter(
            name="A", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, input_size))
        )

    def forward(self, x):
        res = torch.zeros(*x.shape[:-1], device=x.device)
        dres = torch.zeros(*x.shape, device=x.device)
        ddres = torch.zeros(*x.shape, device=x.device)
        for i in range(x.shape[-2]):
            for j in range(self.dim):
                s = 0.0
                for k in range(self.dim):
                    s += x[..., i, k] * self.A[..., k, j]
                    dres[..., i, j] += x[..., i, k] * (self.A[..., j, k] + self.A[..., k, j])
                res[..., i] += x[..., i, j] * s
                ddres[..., i, j] = self.A[..., j, j]
        return torch.cat((res.div(2).unsqueeze(-1), dres.div(2), ddres), -1)
