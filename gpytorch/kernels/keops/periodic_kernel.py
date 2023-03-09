#!/usr/bin/env python3

import torch
from linear_operator.operators import KeOpsLinearOperator
from ... import settings
from .keops_kernel import KeOpsKernel
import math
from typing import Optional
from ...constraints import Interval, Positive
from ...priors import Prior


try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    # subclass from original periodic kernel to reduce code duplication
    class PeriodicKernel(KeOpsKernel):
        """
        Implements the Periodic Kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for gpytorch.kernels.PeriodicKernel in most cases, and supports
        the same arguments.
        """

        has_lengthscale = True

        def __init__(
            self,
            period_length_prior: Optional[Prior] = None,
            period_length_constraint: Optional[Interval] = None,
            **kwargs,
        ):
            super(PeriodicKernel, self).__init__(**kwargs)
            if period_length_constraint is None:
                period_length_constraint = Positive()

            ard_num_dims = kwargs.get("ard_num_dims", 1)
            self.register_parameter(
                name="raw_period_length", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims))
            )

            if period_length_prior is not None:
                if not isinstance(period_length_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(period_length_prior).__name__)
                self.register_prior(
                    "period_length_prior",
                    period_length_prior,
                    lambda m: m.period_length,
                    lambda m, v: m._set_period_length(v),
                )

            self.register_constraint("raw_period_length", period_length_constraint)

        @property
        def period_length(self):
            return self.raw_period_length_constraint.transform(self.raw_period_length)

        @period_length.setter
        def period_length(self, value):
            self._set_period_length(value)

        def _set_period_length(self, value):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.raw_period_length)
            self.initialize(raw_period_length=self.raw_period_length_constraint.inverse_transform(value))

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
                # symbolic array of ..., shape 1 x ndatax2_ x ndim
                x2_ = KEOLazyTensor(x2[..., None, :, :])
                lengthscale = self.lengthscale[..., None, None, 0, :]
                K = (x1_ - x2_).abs().sin().power(2.0).divop(lengthscale).mulop(-2.0).sum(-1).exp()

                return K

        def forward(self, x1, x2, diag=False, **params):

            x1_ = x1.div(self.period_length / math.pi)
            x2_ = x2.div(self.period_length / math.pi)

            covar_func = lambda x1, x2, diag=diag: self.covar_func(x1, x2, diag)

            if diag:
                return covar_func(x1_, x2_, diag=True)

            return KeOpsLinearOperator(x1_, x2_, covar_func)

except ImportError:

    class PeriodicKernel(KeOpsKernel):
        def __init__(self, *args, **kwargs):
            super().__init__()
