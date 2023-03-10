#!/usr/bin/env python3

import math

import torch
from linear_operator.operators import KeOpsLinearOperator

from ... import settings
from ..periodic_kernel import PeriodicKernel as GPeriodicKernel
from .keops_kernel import KeOpsKernel

# from ...kernels import PeriodicKernel gives a cyclic import

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    # subclass from original periodic kernel to reduce code duplication
    class PeriodicKernel(GPeriodicKernel, KeOpsKernel):
        """
        Implements the Periodic Kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for gpytorch.kernels.PeriodicKernel in most cases, and supports
        the same arguments.
        """

        has_lengthscale = True

        # code from the already-implemented Periodic Kernel
        def _nonkeops_covar_func(self, x1, x2, diag=False):

            lengthscale = self.lengthscale

            # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
            diff = self.covar_dist(x1, x2, diag=diag, last_dim_is_batch=True)

            if diag:
                lengthscale = lengthscale[..., 0, :, None]
            else:
                lengthscale = lengthscale[..., 0, :, None, None]

            exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

            return exp_term.exp()

        def covar_func(self, x1, x2, diag=False):
            # We only should use KeOps on big kernel matrices
            # If we would otherwise be performing Cholesky inference, (or when just computing a kernel matrix diag)
            # then don't apply KeOps
            # enable gradients to ensure that test time caches on small predictions are still
            # backprop-able
            with torch.autograd.enable_grad():
                if (
                    diag
                    or x1.size(-2) < settings.max_cholesky_size.value()
                    or x2.size(-2) < settings.max_cholesky_size.value()
                ):
                    return self._nonkeops_covar_func(x1, x2, diag=diag)

                # symbolic array of shape ..., ndatax1_ x 1 x ndim
                x1_ = KEOLazyTensor(x1[..., :, None, :])
                # symbolic array of shape ..., 1 x ndatax2_ x ndim
                x2_ = KEOLazyTensor(x2[..., None, :, :])
                lengthscale = self.lengthscale[..., None, None, 0, :]
                # do not use .power(2.0) as it gives NaN values on cuda
                # seems related to https://github.com/getkeops/keops/issues/112
                K = ((((x1_ - x2_).abs().sin()) ** 2) * (-2.0 / lengthscale)).sum(-1).exp()

                return K

        def forward(self, x1, x2, diag=False, **params):

            x1_ = x1.div(self.period_length / math.pi)
            x2_ = x2.div(self.period_length / math.pi)

            covar_func = lambda x1, x2, diag=diag: self.covar_func(x1, x2, diag)

            if diag:
                return covar_func(x1_, x2_, diag=True)

            return KeOpsLinearOperator(x1_, x2_, covar_func)

        # taken from KeOpsKernel, otherwise Kernel.__call__ will be used directly
        def __call__(self, *args, **kwargs):
            # Hotfix for zero gradients. See https://github.com/cornellius-gp/gpytorch/issues/1543
            args = [arg.contiguous() if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.contiguous() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return super().__call__(*args, **kwargs)

except ImportError:

    class PeriodicKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()
