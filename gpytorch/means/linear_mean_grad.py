#!/usr/bin/env python3

import torch

from .mean import Mean


class LinearMeanGrad(Mean):
    r"""
    A linear prior mean function and its first derivative, i.e.:

    .. math::

        \mu(\mathbf x) &= \mathbf W \cdot \mathbf x + B \\
        \nabla \mu(\mathbf x) &= \mathbf W

    where :math:`\mathbf W` and :math:`B` are learned constants.

    :param input_size: dimension of input :math:`\mathbf x`.
    :type input_size: int
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional
    :param bias: True/False flag for whether the bias: :math:`B` should be used in the mean (default: True).
    :type bias: bool, optional

    :var torch.Tensor weights: :math:`\mathbf W` parameter
    :var torch.Tensor bias: :math:`B` parameter
    """

    def __init__(self, input_size: int, batch_shape: torch.Size = torch.Size(), bias: bool = True):
        super().__init__()
        self.dim = input_size
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights)
        if self.bias is not None:
            res = res + self.bias.unsqueeze(-1)
        dres = self.weights.expand(x.transpose(-1, -2).shape).transpose(-1, -2)
        return torch.cat((res, dres), -1)
