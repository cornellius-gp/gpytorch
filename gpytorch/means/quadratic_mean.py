#!/usr/bin/env python3

import torch

from .mean import Mean


class QuadraticMean(Mean):
    def __init__(self, input_size: int, batch_shape: torch.Size = torch.Size()):
        super().__init__()
        self.dim = input_size
        self.batch_shape = batch_shape
        self.register_parameter(
            name="cholesky", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size * (input_size + 1) // 2))
        )

    def forward(self, x):
        xl = torch.zeros(*x.shape, device=x.device)
        for i in range(x.shape[-2]):
            for j in range(self.dim):
                for k in range(j, self.dim):
                    xl[..., i, j] += self.cholesky[..., k * (k + 1) // 2 + j] * x[..., i, k]
        res = xl.pow(2).sum(-1)
        return res
