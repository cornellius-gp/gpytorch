#!/usr/bin/env python3

import torch

from ..utils.broadcasting import _mul_broadcast_shape
from .mean import Mean


class ConstantMeanGrad(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super(ConstantMeanGrad, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        batch_shape = _mul_broadcast_shape(self.batch_shape, input.shape[:-2])
        mean = self.constant.unsqueeze(-1).expand(*batch_shape, input.size(-2), input.size(-1) + 1).contiguous()
        mean[..., 1:] = 0
        return mean
