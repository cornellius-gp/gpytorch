#!/usr/bin/env python3

import torch
from .mean import Mean


class ConstantMean(Mean):
    def __init__(self, prior=None, batch_size=None):
        super(ConstantMean, self).__init__()
        self.batch_size = batch_size
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(batch_size or 1, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        return self.constant.expand(input.size(0), input.size(1))
