#!/usr/bin/env python3

import torch

from ..lazy import DiagLazyTensor, KroneckerProductLazyTensor
from .kernel import Kernel


class GPFAKernel(Kernel):
    r"""
    Kernel supporting Kronecker style multitask Gaussian processes (where every data point is evaluated at every
    task) using :class:`gpytorch.kernels.IndexKernel` as a basic multitask kernel.

    Given a base covariance module to be used for the data, :math:`K_{XX}`, this kernel computes a task kernel of
    specified size :math:`K_{TT}` and returns :math:`K = K_{TT} \otimes K_{XX}`. as an
    :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the data kernel.
    :param int num_tasks: Number of tasks
    :param int rank: (default 1) Rank of index kernel to use for task covariance matrix.
    :param ~gpytorch.priors.Prior task_covar_prior: (default None) Prior to use for task kernel.
        See :class:`gpytorch.kernels.IndexKernel` for details.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """
    """
    TODO(jasmine):deal with docstrings and comments
    input dim is input dim at each timepoint, e.g. if just putting in

    need num_tasks = = len(data_covar_modules)
    """
    # TODO: confirm if don't need priors / constraints on most of this.
    def __init__(
        self,
        latent_covar_module,
        num_latents,
        num_obs,  # GPFA_component = Reversible_GPFA_Component_Kernel,
        # C_prior=None, C_constraint=None, R_prior = None,
        **kwargs,
    ):
        super(GPFAKernel, self).__init__(**kwargs)
        self.num_obs = num_obs
        self.num_latents = num_latents
        # gpfa_components = [GPFA_component(data_covar_modules[i],num_latents,i) for i in range(num_latents)]
        # TODO: check for length of data_covar modules matching, or length one, in which case repeat the same module,
        # deep copy, like in multitaskmean
        self.covar_module = latent_covar_module
        # AdditiveKernel(*[GPFA_component(data_covar_modules[i],num_latents,i) for i in range(num_latents)])
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
        covar = kron_prod @ self.covar_module(x1, x2, **params) @ kron_prod.t()
        return covar.diag() if diag else covar

    def num_outputs_per_input(self, x1, x2):
        """`
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_obs) x (m*num_obs)` covariance matrix.
        """
        return self.num_obs
