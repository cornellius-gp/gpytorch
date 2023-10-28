#!/usr/bin/env python3

import torch

from .mean import Mean


class QuadraticMean(Mean):
    r"""
    A quadratic prior mean function, i.e.:

    .. math::

        \mu(\mathbf x) &= \frac12 \mathbf x^\top \cdot A \cdot \mathbf x

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
        res = torch.zeros(*x.shape[:-1])
        for i in range(x.shape[-2]):
            for j in range(self.dim):
                s = 0.0
                for k in range(self.dim):
                    s += x[..., i, k] * self.A[..., k, j]
                res[..., i] += x[..., i, j] * s
        return res.div(2)
