#!/usr/bin/env python3

from typing import Any, Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from .. import settings
from ..constraints import GreaterThan
from ..distributions import MultivariateNormal
from ..lazy import ConstantDiagLazyTensor, DiagLazyTensor, ZeroLazyTensor
from ..module import Module
from ..utils.broadcasting import _mul_broadcast_shape


class Noise(Module):
    pass


class _HomoskedasticNoiseBase(Noise):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), num_tasks=1):
        super().__init__()
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        self.register_parameter(name="raw_noise", parameter=Parameter(torch.zeros(*batch_shape, num_tasks)))
        if noise_prior is not None:
            self.register_prior("noise_prior", noise_prior, lambda m: m.noise, lambda m, v: m._set_noise(v))

        self.register_constraint("raw_noise", noise_constraint)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    @noise.setter
    def noise(self, value: Tensor) -> None:
        self._set_noise(value)

    def _set_noise(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def forward(self, *params: Any, shape: Optional[torch.Size] = None, **kwargs: Any) -> DiagLazyTensor:
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

        If a "noise" kwarg (a Tensor) is provided, this noise is used directly.
        """
        if "noise" in kwargs:
            return DiagLazyTensor(kwargs.get("noise"))
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]
        noise = self.noise
        *batch_shape, n = shape
        noise_batch_shape = noise.shape[:-1] if noise.dim() > 1 else torch.Size()
        num_tasks = noise.shape[-1]
        batch_shape = _mul_broadcast_shape(noise_batch_shape, batch_shape)
        noise = noise.unsqueeze(-2)
        noise_diag = noise.expand(*batch_shape, 1, num_tasks).contiguous()
        if num_tasks == 1:
            noise_diag = noise_diag.view(*batch_shape, 1)
        if noise_diag.shape[-1] != 1:
            noise_diag = noise_diag.unsqueeze(-1)
        return ConstantDiagLazyTensor(noise_diag, diag_shape=n)


class HomoskedasticNoise(_HomoskedasticNoiseBase):
    def __init__(self, noise_prior=None, noise_constraint=None, batch_shape=torch.Size()):
        super().__init__(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape, num_tasks=1
        )


class MultitaskHomoskedasticNoise(_HomoskedasticNoiseBase):
    def __init__(self, num_tasks, noise_prior=None, noise_constraint=None, batch_shape=torch.Size()):
        super().__init__(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape, num_tasks=num_tasks
        )


class HeteroskedasticNoise(Noise):
    def __init__(self, noise_model, noise_indices=None, noise_constraint=None):
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)
        super().__init__()
        self.noise_model = noise_model
        self._noise_constraint = noise_constraint
        self._noise_indices = noise_indices

    def forward(
        self,
        *params: Any,
        batch_shape: Optional[torch.Size] = None,
        shape: Optional[torch.Size] = None,
        noise: Optional[Tensor] = None,
    ) -> DiagLazyTensor:
        if noise is not None:
            return DiagLazyTensor(noise)
        training = self.noise_model.training  # keep track of mode
        self.noise_model.eval()  # we want the posterior prediction of the noise model
        with settings.detach_test_caches(False), settings.debug(False):
            if len(params) == 1 and not torch.is_tensor(params[0]):
                output = self.noise_model(*params[0])
            else:
                output = self.noise_model(*params)
        self.noise_model.train(training)
        if not isinstance(output, MultivariateNormal):
            raise NotImplementedError("Currently only noise models that return a MultivariateNormal are supported")
        # note: this also works with MultitaskMultivariateNormal, where this
        # will return a batched DiagLazyTensors of size n x num_tasks x num_tasks
        noise_diag = output.mean if self._noise_indices is None else output.mean[..., self._noise_indices]
        return DiagLazyTensor(self._noise_constraint.transform(noise_diag))


class FixedGaussianNoise(Module):
    def __init__(self, noise: Tensor) -> None:
        super().__init__()
        self.noise = noise

    def forward(
        self, *params: Any, shape: Optional[torch.Size] = None, noise: Optional[Tensor] = None, **kwargs: Any
    ) -> DiagLazyTensor:
        if shape is None:
            p = params[0] if torch.is_tensor(params[0]) else params[0][0]
            shape = p.shape if len(p.shape) == 1 else p.shape[:-1]

        if noise is not None:
            return DiagLazyTensor(noise)
        elif shape[-1] == self.noise.shape[-1]:
            return DiagLazyTensor(self.noise)
        else:
            return ZeroLazyTensor()

    def _apply(self, fn):
        self.noise = fn(self.noise)
        return super(FixedGaussianNoise, self)._apply(fn)
