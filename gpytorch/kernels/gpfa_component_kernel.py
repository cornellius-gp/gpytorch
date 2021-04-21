#!/usr/bin/env python3

import torch

from ..lazy import DiagLazyTensor, KroneckerProductLazyTensor, lazify
from .kernel import Kernel


class GPFAComponentKernel(Kernel):
    r"""
     Kernel supporting Gaussian Process Factor Analysis using
     :class:`gpytorch.kernels.GPFAComponentKernel` as a basic GPFA latent kernel.


    Given a base covariance module to be used for a latent, :math:`K_{XX}`, this kernel computes a latent kernel of
    specified size :math:`K_MM}` that is zeros everywhere except :math:`K_{kernel_loc,kernel_loc}` and returns
    :math:`K = K_{MM} \otimes K_{XX}`. as an :obj:`gpytorch.lazy.KroneckerProductLazyTensor`.

    :param ~gpytorch.kernels.Kernel data_covar_module: Kernel to use as the latent kernel.
    :param int num_latents: Number of latents (M)
    :param int kernel_loc: Latent number that this kernel represents.
    :param dict kwargs: Additional arguments to pass to the kernel.
    """

    def __init__(self, data_covar_module, num_latents, kernel_loc, **kwargs):
        """
        """
        super(GPFAComponentKernel, self).__init__(**kwargs)
        task_diag = torch.zeros(num_latents)
        task_diag[kernel_loc] = 1
        self.task_covar = DiagLazyTensor(task_diag)
        self.data_covar_module = data_covar_module
        self.num_latents = num_latents

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("GPFAComponentKernel does not accept the last_dim_is_batch argument.")
        covar_i = self.task_covar
        if len(x1.shape[:-2]):
            covar_i = covar_i.repeat(*x1.shape[:-2], 1, 1)
        covar_x = lazify(self.data_covar_module.forward(x1, x2, **params))
        res = KroneckerProductLazyTensor(covar_x, covar_i)
        return res.diag() if diag else res

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this
        kernel returns an `(n*num_latents) x (m*num_latents)` covariance matrix.
        """
        return self.num_latents
