#!/usr/bin/env python3

import math

from linear_operator.operators import KernelLinearOperator

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
        def _nonkeops_covar_func(self, x1, x2, lengthscale, diag=False):
            # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
            diff = self.covar_dist(x1, x2, diag=diag, last_dim_is_batch=True)

            if diag:
                lengthscale = lengthscale[..., 0, :, None]
            else:
                lengthscale = lengthscale[..., 0, :, None, None]

            exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

            return exp_term.exp()

        def covar_func(self, x1, x2, lengthscale, **kwargs):
            if x1.size(-2) < settings.max_cholesky_size.value() or x2.size(-2) < settings.max_cholesky_size.value():
                return self._nonkeops_covar_func(x1, x2, lengthscale)

            # symbolic array of shape ..., ndatax1_ x 1 x ndim
            x1_ = KEOLazyTensor(x1[..., :, None, :])
            # symbolic array of shape ..., 1 x ndatax2_ x ndim
            x2_ = KEOLazyTensor(x2[..., None, :, :])
            lengthscale = lengthscale[..., None, None, 0, :]  # 1 x 1 x ndim
            # do not use .power(2.0) as it gives NaN values on cuda
            # seems related to https://github.com/getkeops/keops/issues/112
            K = ((((x1_ - x2_).abs().sin()) ** 2) * (-2.0 / lengthscale)).sum(-1).exp()

            return K

        def forward(self, x1, x2, diag=False, **kwargs):
            x1_ = x1.div(self.period_length / math.pi)
            x2_ = x2.div(self.period_length / math.pi)

            if diag:
                return self._nonkeops_covar_func(x1_, x2_, self.lengthscale, diag=diag)

            # return KernelLinearOperator inst only when calculating the whole covariance matrix
            # pass any parameters which are used inside _covar_func as *args to get gradients computed for them
            return KernelLinearOperator(x1_, x2_, lengthscale=self.lengthscale, covar_func=self.covar_func, **kwargs)

except ImportError:

    class PeriodicKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
