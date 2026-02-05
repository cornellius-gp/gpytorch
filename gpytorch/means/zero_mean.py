#!/usr/bin/env python3

from __future__ import annotations

import torch

from .mean import Mean


class ZeroMean(Mean):
    def __init__(self, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        mean = torch.zeros(*self.batch_shape, 1, dtype=input.dtype, device=input.device)
        if input.shape[:-2] == self.batch_shape:
            return mean.expand(input.shape[:-1])
        else:
            return mean.expand(torch.broadcast_shapes(input.shape[:-1], mean.shape))
