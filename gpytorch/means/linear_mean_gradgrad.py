#!/usr/bin/env python3

import torch

from .mean import Mean


class LinearMeanGradGrad(Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
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
        dres = self.weights.expand(x.transpose(-1,-2).shape).transpose(-1,-2)
        ddres = torch.zeros_like(dres)
        return torch.cat((res,dres,ddres),-1)
