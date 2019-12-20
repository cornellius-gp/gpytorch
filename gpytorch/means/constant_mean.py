#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _mul_broadcast_shape
from .mean import Mean


class ConstantMean(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super(ConstantMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        if input.shape[:-2] == self.batch_shape:
            return self.constant.expand(input.shape[:-1])
        else:
            return self.constant.expand(_mul_broadcast_shape(input.shape[:-1], self.constant.shape))
