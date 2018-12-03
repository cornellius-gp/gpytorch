#!/usr/bin/env python3

import torch
from torch.nn import Parameter
from torch.nn.functional import softplus

from ..distributions import MultivariateNormal
from ..lazy import DiagLazyTensor
from ..module import Module
from ..utils.broadcasting import _mul_broadcast_shape
from ..utils.transforms import _get_inv_param_transform


class _HomoskedasticNoiseBase(Module):
    def __init__(self, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None, num_tasks=1):
        super().__init__()
        self._param_transform = param_transform
        self._inv_param_transform = _get_inv_param_transform(param_transform, inv_param_transform)
        self.register_parameter(name="raw_noise", parameter=Parameter(torch.zeros(batch_size, num_tasks)))
        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda: self.noise, lambda v: self._set_noise(v))

    @property
    def noise(self):
        return self._param_transform(self.raw_noise)

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    def _set_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(raw_noise=self._inv_param_transform(value))

    def forward(self, *params, shape=None):
        """In the homoskedastic case, the parameters are only used to infer the required shape.
        Here are the possible scenarios:
        - non-batched noise, non-batched input, non-MT -> noise_diag shape is `n`
        - non-batched noise, non-batched input, MT -> noise_diag shape is `nt`
        - non-batched noise, batched input, non-MT -> noise_diag shape is `b x n` with b' the broadcasted batch shape
        - non-batched noise, batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, non-batched input, non-MT -> noise_diag shape is `b x n`
        - batched noise, non-batched input, MT -> noise_diag shape is `b x nt`
        - batched noise, batched input, non-MT -> noise_diag shape is `b' x n`
        - batched noise, batched input, MT -> noise_diag shape is `b' x nt`
        where `n` is the number of evaluation points and `t` is the number of tasks (i.e. `num_tasks` of self.noise).
        So bascially the shape is always `b' x nt`, with `b'` appropriately broadcast from the noise parameter and
        input batch shapes. `n` and the input batch shape are determined either from the shape arg or from the params
        input. For this it is sufficient to take in a single `shape` arg, with the convention that shape[:-1] is the
        batch shape of the input, and shape[-1] is `n`.
        """
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        noise = self.noise
        batch_shape, n = shape[:-1], shape[-1]
        noise_batch_shape = noise.shape[:-1] if noise.shape[-2] > 1 else torch.Size()
        num_tasks = noise.shape[-1]
        batch_shape = _mul_broadcast_shape(noise_batch_shape, batch_shape)
        noise = noise.unsqueeze(-2)
        if len(batch_shape) == 0:
            noise = noise.squeeze(0)
        noise_diag = noise.expand(batch_shape + torch.Size([n, num_tasks])).contiguous()
        if num_tasks == 1:
            noise_diag = noise_diag.view(*batch_shape, n)
        return DiagLazyTensor(noise_diag)


class HomoskedasticNoise(_HomoskedasticNoiseBase):
    def __init__(self, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None):
        super().__init__(
            noise_prior=noise_prior,
            batch_size=batch_size,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
            num_tasks=1,
        )


class MultitaskHomoskedasticNoise(_HomoskedasticNoiseBase):
    def __init__(self, num_tasks, noise_prior=None, batch_size=1, param_transform=softplus, inv_param_transform=None):
        super().__init__(
            noise_prior=noise_prior,
            batch_size=batch_size,
            param_transform=param_transform,
            inv_param_transform=inv_param_transform,
            num_tasks=num_tasks,
        )


class HeteroskedasticNoise(Module):
    def __init__(self, noise_model, noise_indices=None, noise_transform=torch.exp):
        super().__init__()
        self.noise_model = noise_model
        self._noise_transform = noise_transform
        self._noise_indices = noise_indices
        self._noise_transform = noise_transform

    def forward(self, *params, batch_shape=None, shape=None):
        if len(params) == 1 and not torch.is_tensor(params[0]):
            output = self.noise_model(*params[0])
        else:
            output = self.noise_model(*params)
        if not isinstance(output, MultivariateNormal):
            raise NotImplementedError("Currently only noise models that return a MultivariateNormal are supported")
        # note: this also works with MultitaskMultivariateNormal, where this
        # will return a batched DiagLazyTensors of size n x num_tasks x num_tasks
        noise_diag = output.mean if self._noise_indices is None else output.mean[..., self._noise_indices]
        return DiagLazyTensor(self._noise_transform(noise_diag))
