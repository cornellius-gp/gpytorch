#!/usr/bin/env python3
import math

import torch
from linear_operator.operators import KernelLinearOperator

from ..matern_kernel import MaternKernel as GMaternKernel
from .keops_kernel import KeOpsKernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    def _covar_func(x1, x2, nu=2.5, **params):
        x1_ = KEOLazyTensor(x1[..., :, None, :])
        x2_ = KEOLazyTensor(x2[..., None, :, :])

        distance = ((x1_ - x2_) ** 2).sum(-1).sqrt()
        exp_component = (-math.sqrt(nu * 2) * distance).exp()

        if nu == 0.5:
            constant_component = 1
        elif nu == 1.5:
            constant_component = (math.sqrt(3) * distance) + 1
        elif nu == 2.5:
            constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * (distance**2))

        return constant_component * exp_component

    class MaternKernel(KeOpsKernel):
        """
        Implements the Matern kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for :class:`gpytorch.kernels.MaternKernel` in most cases,
        and supports the same arguments.

        :param nu: (Default: 2.5) The smoothness parameter.
        :type nu: float (0.5, 1.5, or 2.5)
        :param ard_num_dims: (Default: `None`) Set this if you want a separate lengthscale for each
            input dimension. It should be `d` if x1 is a `... x n x d` matrix.
        :type ard_num_dims: int, optional
        :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
             batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
        :type batch_shape: torch.Size, optional
        :param active_dims: (Default: `None`) Set this if you want to
            compute the covariance of only a few input dimensions. The ints
            corresponds to the indices of the dimensions.
        :type active_dims: Tuple(int)
        :param lengthscale_prior: (Default: `None`)
            Set this if you want to apply a prior to the lengthscale parameter.
        :type lengthscale_prior: ~gpytorch.priors.Prior, optional
        :param lengthscale_constraint: (Default: `Positive`) Set this if you want
            to apply a constraint to the lengthscale parameter.
        :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
        :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
        :type eps: float, optional
        """

        has_lengthscale = True

        def __init__(self, nu=2.5, **kwargs):
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
            super(MaternKernel, self).__init__(**kwargs)
            self.nu = nu

        def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
            x1_ = (x1 - mean) / self.lengthscale
            x2_ = (x2 - mean) / self.lengthscale

            distance = self.covar_dist(x1_, x2_, diag=diag, **kwargs)
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance**2)
            return constant_component * exp_component

        def _keops_forward(self, x1, x2, **kwargs):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
            x1_ = (x1 - mean) / self.lengthscale
            x2_ = (x2 - mean) / self.lengthscale
            # return KernelLinearOperator inst only when calculating the whole covariance matrix
            return KernelLinearOperator(x1_, x2_, covar_func=_covar_func, nu=self.nu, **kwargs)

except ImportError:

    class MaternKernel(GMaternKernel):
        pass
