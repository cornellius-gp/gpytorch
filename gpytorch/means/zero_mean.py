#!/usr/bin/env python3

import torch
from .mean import Mean


class ZeroMean(Mean):
    def forward(self, input):
        return torch.zeros((input.size(0), input.size(1)), dtype=input.dtype, device=input.device)
