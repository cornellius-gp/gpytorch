from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.nn import Parameter

from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor
from ..module import Module


class HomoskedasticNoise(Module):
    def __init__(self, log_noise_prior=None, batch_size=1):
        super(HomoskedasticNoise, self).__init__()
        self.register_parameter(
            name="log_noise", parameter=Parameter(torch.zeros(batch_size, 1)), prior=log_noise_prior
        )

    def forward(self, params):
        log_noise = self.log_noise
        p = params[0] if isinstance(params, list) else params
        var_shape = p.shape[:-2] + p.shape[-1:]
        if len(var_shape) == 1:
            log_noise = log_noise.squeeze(0)
        variances = log_noise * torch.ones(*var_shape, dtype=log_noise.dtype, device=log_noise.device)
        return DiagLazyTensor(variances)


class HeteroskedasticNoise(Module):
    def __init__(self, log_noise_model):
        super(HeteroskedasticNoise, self).__init__()
        self.log_noise_model = log_noise_model

    def forward(self, params):
        output = self.log_noise_model(params[0] if isinstance(params, list) or isinstance(params, tuple) else params)
        if not isinstance(output, MultivariateNormal):
            raise NotImplementedError("Currently only log-noise models that return a MultivariateNormal are supported")
        # note: this also works with MultitaskMultivariateNormal, where this
        # will return a batched DiagLazyTensors of size n x num_tasks x num_tasks
        return DiagLazyTensor(output.mean)
