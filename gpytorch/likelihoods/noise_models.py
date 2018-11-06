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
        log_noise = self.log_noise.squeeze(0).exp()
        p = params[0] if isinstance(params, list) else params
        shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        if log_noise.ndimension() > len(shape):
            raise RuntimeError("Must provide batched input if in batch mode")
        if log_noise.shape[-1] > 1:  # deal with multi-task case
            shape = shape + torch.Size([log_noise.shape[-1]])
        log_noise_diag = log_noise.expand(shape)
        return DiagLazyTensor(log_noise_diag)


class MultitaskHomoskedasticNoise(HomoskedasticNoise):
    def __init__(self, num_tasks, log_noise_prior=None, batch_size=1):
        super(HomoskedasticNoise, self).__init__()
        self.register_parameter(
            name="log_noise", parameter=Parameter(torch.zeros(batch_size, num_tasks)), prior=log_noise_prior
        )


class HeteroskedasticNoise(Module):
    def __init__(self, noise_model, log_scale=True, noise_indices=None):
        super(HeteroskedasticNoise, self).__init__()
        self.noise_model = noise_model
        self.log_scale = log_scale
        self.noise_indices = noise_indices

    def forward(self, params):
        output = self.noise_model(params[0] if isinstance(params, list) or isinstance(params, tuple) else params)
        if not isinstance(output, MultivariateNormal):
            raise NotImplementedError("Currently only noise models that return a MultivariateNormal are supported")
        # note: this also works with MultitaskMultivariateNormal, where this
        # will return a batched DiagLazyTensors of size n x num_tasks x num_tasks
        noise_diag = output.mean if self.noise_indices is None else output.mean[..., self.noise_indices]
        return DiagLazyTensor(noise_diag.exp() if self.log_scale else noise_diag)
