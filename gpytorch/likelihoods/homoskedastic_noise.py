from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.nn import Parameter

from ..lazy import DiagLazyTensor
from ..module import Module


class HomoskedasticNoise(Module):
    def __init__(self, log_noise_prior=None, batch_size=1):
        super(HomoskedasticNoise, self).__init__()
        self.register_parameter(
            name="log_noise", parameter=Parameter(torch.zeros(batch_size, 1)), prior=log_noise_prior
        )

    def forward(self, params):
        noise = self.log_noise.exp()
        if isinstance(params, list):
            variance_shape = params[0].shape[:-2] + params[0].shape[-1:]
        else:
            variance_shape = params.shape[:-2] + params.shape[-1:]
        if len(variance_shape) == 1:
            noise = noise.squeeze(0)
        variances = noise * torch.ones(*variance_shape, dtype=noise.dtype, device=noise.device)
        return DiagLazyTensor(variances)
