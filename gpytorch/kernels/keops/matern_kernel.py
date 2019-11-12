#!/usr/bin/env python3
import math

import torch

from ...lazy import KeOpsLazyTensor
from .keops_kernel import KeOpsKernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class MaternKernel(KeOpsKernel):
        """
        Implements the Matern kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for gpytorch.kernels.MaternKernel in most cases, and supports
        the same arguments. There are currently a few limitations, for example a lack of batch mode support. However,
        most other features like ARD will work.
        """

        has_lengthscale = True

        def __init__(self, nu=2.5, **kwargs):
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
            super(MaternKernel, self).__init__(**kwargs)
            self.nu = nu

        def covar_func(self, x1, x2, diag=False):
            # TODO: x1 / x2 size checks are a work around for a very minor bug in KeOps.
            # This bug is fixed on KeOps master, and we'll remove that part of the check
            # when they cut a new release.
            if diag or x1.size(-2) == 1 or x2.size(-2) == 1:
                distance = self.covar_dist(x1, x2, diag=diag)
                exp_component = torch.exp(-math.sqrt(self.nu * 2) * distance)

                if self.nu == 0.5:
                    constant_component = 1
                elif self.nu == 1.5:
                    constant_component = (math.sqrt(3) * distance).add(1)
                elif self.nu == 2.5:
                    constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
                return constant_component * exp_component
            else:
                with torch.autograd.enable_grad():
                    x1_ = KEOLazyTensor(x1[..., :, None, :])
                    x2_ = KEOLazyTensor(x2[..., None, :, :])

                    distance = ((x1_ - x2_) ** 2).sum(-1).sqrt()
                    exp_component = (-math.sqrt(self.nu * 2) * distance).exp()

                    if self.nu == 0.5:
                        constant_component = 1
                    elif self.nu == 1.5:
                        constant_component = (math.sqrt(3) * distance) + 1
                    elif self.nu == 2.5:
                        constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * distance ** 2)

                    return constant_component * exp_component

        def forward(self, x1, x2, diag=False, **params):
            mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]

            x1_ = (x1 - mean).div(self.lengthscale)
            x2_ = (x2 - mean).div(self.lengthscale)

            if diag:
                return self.covar_func(x1_, x2_, diag=True)

            covar_func = lambda x1, x2, diag=False: self.covar_func(x1, x2, diag)
            return KeOpsLazyTensor(x1_, x2_, covar_func)


except ImportError:

    class MaternKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()
