#!/usr/bin/env python3

# from linear_operator.operators import KeOpsLinearOperator
from linear_operator.operators import KernelLinearOperator

from ... import settings
from ..rbf_kernel import postprocess_rbf
from .keops_kernel import KeOpsKernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class RBFKernel(KeOpsKernel):
        """
        Implements the RBF kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for gpytorch.kernels.RBFKernel in most cases, and supports
        the same arguments.
        """

        has_lengthscale = True

        def _nonkeops_covar_func(self, x1, x2, diag=False):
            return postprocess_rbf(self.covar_dist(x1, x2, square_dist=True, diag=diag))

        def covar_func(self, x1, x2, **kwargs):
            if x1.size(-2) < settings.max_cholesky_size.value() or x2.size(-2) < settings.max_cholesky_size.value():
                return self._nonkeops_covar_func(x1, x2)

            x1_ = KEOLazyTensor(x1[..., :, None, :])
            x2_ = KEOLazyTensor(x2[..., None, :, :])

            K = (-((x1_ - x2_) ** 2).sum(-1) / 2).exp()

            return K

        def forward(self, x1, x2, diag=False, **kwargs):

            x1_ = x1 / self.lengthscale
            x2_ = x2 / self.lengthscale

            if diag:
                return self._nonkeops_covar_func(x1_, x2_, diag=diag)

            # return KernelLinearOperator inst only when calculating the whole covariance matrix
            return KernelLinearOperator(x1_, x2_, covar_func=self.covar_func, **kwargs)

except ImportError:

    class RBFKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
