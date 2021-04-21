#!/usr/bin/env python3

from copy import deepcopy

import torch

from ..lazy import DiagLazyTensor, KroneckerProductLazyTensor
from .gpfa_component_kernel import GPFAComponentKernel
from .kernel import AdditiveKernel, Kernel


class GPFAKernel(Kernel):
    r"""
    Kernel supporting Gaussian Process Factor Analysis using
    :class:`gpytorch.kernels.GPFAComponentKernel` as a basic GPFA latent kernel.

    Given base covariance modules to be used for the latents, :math:`k_i`, this kernel
    puts the base covariance modules in a block diagonal with :math:`M` blocks as :math:`K_{XX}`.
    This defines :math:`C \in MxN` and returns :math:`(I_T \otimes C)K_{XX}(I_T \otimes C)^T` as an
    :obj:`gpytorch.lazy.LazyEvaluatedKernelTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the latent kernel.
    :param int num_latents: Number of latents (M).
    :param int num_obs: Number of observation dimensions (typically, the number of neurons, N).
    :param ~gpytorch.kernels.Kernel GPFA_component: (default GPFAComponentKernel) Kernel to use to scale the latent
    kernels to the necessary shape.
    GPFAComponentKernel is currently the only option; if non-reversible kernels are later added,
    there will then be another option here.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """
    # TODO: confirm if don't need priors / constraints on most of this.
    def __init__(
        self,
        data_covar_modules,
        num_latents,
        num_obs,
        GPFA_component=GPFAComponentKernel,
        # C_prior=None, C_constraint=None,
        **kwargs,
    ):
        super(GPFAKernel, self).__init__(**kwargs)
        self.num_obs = num_obs
        self.num_latents = num_latents

        if len(data_covar_modules) == 1:
            data_covar_modules = data_covar_modules + [
                deepcopy(data_covar_modules[0]) for i in range(num_latents - 1)
            ]  # TODO test this line
        self.latent_covar_module = AdditiveKernel(
            *[GPFA_component(data_covar_modules[i], num_latents, i) for i in range(num_latents)]
        )
        self.register_parameter(name="raw_C", parameter=torch.nn.Parameter(torch.randn(num_obs, num_latents)))

    @property
    def C(self):
        return self.raw_C

    @C.setter
    def C(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_C)

        self.initialize(raw_C=value)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("GPFAKernel does not yet accept the last_dim_is_batch argument.")
        # what to do with I_t if have dif numbers of input & output timesteps? TODO (jasmine): fix this
        I_t = DiagLazyTensor(torch.ones(len(x1)))
        kron_prod = KroneckerProductLazyTensor(I_t, self.C)
        covar = kron_prod @ self.latent_covar_module(x1, x2, **params) @ kron_prod.t()
        return covar.diag() if diag else covar

    def num_outputs_per_input(self, x1, x2):
        """`
        Given `n` data points `x1` and `m` datapoints `x2`, this
        kernel returns an `(n*num_obs) x (m*num_obs)` covariance matrix.
        """
        return self.num_obs
