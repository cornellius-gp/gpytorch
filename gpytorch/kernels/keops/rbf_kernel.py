#!/usr/bin/env python3

import torch
from linear_operator.operators import KeOpsLinearOperator

from ... import settings
from ..rbf_kernel import postprocess_rbf
from .keops_kernel import KeOpsKernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class RBFKernel(KeOpsKernel):
        """
        Implements the RBF kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for gpytorch.kernels.RBFKernel in most cases, and supports
        the same arguments. There are currently a few limitations, for example a lack of batch mode support. However,
        most other features like ARD will work.
        """

        has_lengthscale = True

        def _nonkeops_covar_func(self, x1, x2, diag=False):
            return postprocess_rbf(
                self.covar_dist(x1, x2, square_dist=True, diag=diag)
            )

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

                x1_ = KEOLazyTensor(x1[..., :, None, :])
                x2_ = KEOLazyTensor(x2[..., None, :, :])

                K = (-((x1_ - x2_) ** 2).sum(-1) / 2).exp()

                return K

        def forward(self, x1, x2, diag=False, **params):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)

            covar_func = lambda x1, x2, diag=diag: self.covar_func(x1, x2, diag)

            if diag:
                return covar_func(x1_, x2_, diag=True)

            return KeOpsLinearOperator(x1_, x2_, covar_func)

except ImportError:

    class RBFKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()
