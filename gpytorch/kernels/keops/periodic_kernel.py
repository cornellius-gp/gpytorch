#!/usr/bin/env python3

import math

from linear_operator.operators import KernelLinearOperator

from ..periodic_kernel import PeriodicKernel as GPeriodicKernel
from .keops_kernel import KeOpsKernel

# from ...kernels import PeriodicKernel gives a cyclic import

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    def _covar_func(x1, x2, lengthscale, **kwargs):
        # symbolic array of shape ..., ndatax1_ x 1 x ndim
        x1_ = KEOLazyTensor(x1[..., :, None, :])
        # symbolic array of shape ..., 1 x ndatax2_ x ndim
        x2_ = KEOLazyTensor(x2[..., None, :, :])
        lengthscale = lengthscale[..., None, None, 0, :]  # 1 x 1 x ndim
        # do not use .power(2.0) as it gives NaN values on cuda
        # seems related to https://github.com/getkeops/keops/issues/112
        K = ((((x1_ - x2_).abs().sin()) ** 2) * (-2.0 / lengthscale)).sum(-1).exp()
        return K

    # subclass from original periodic kernel to reduce code duplication
    class PeriodicKernel(KeOpsKernel, GPeriodicKernel):
        """
        Implements the Periodic Kernel using KeOps as a driver for kernel matrix multiplies.

        This class can be used as a drop in replacement for :class:`gpytorch.kernels.PeriodicKernel` in most cases,
        and supports the same arguments.

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
        :param period_length_prior: (Default: `None`)
            Set this if you want to apply a prior to the period length parameter.
        :type period_length_prior: ~gpytorch.priors.Prior, optional
        :param period_length_constraint: (Default: `Positive`) Set this if you want
            to apply a constraint to the period length parameter.
        :type period_length_constraint: ~gpytorch.constraints.Interval, optional
        :param lengthscale_prior: (Default: `None`)
            Set this if you want to apply a prior to the lengthscale parameter.
        :type lengthscale_prior: ~gpytorch.priors.Prior, optional
        :param lengthscale_constraint: (Default: `Positive`) Set this if you want
            to apply a constraint to the lengthscale parameter.
        :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
        :param eps: (Default: 1e-6) The minimum value that the lengthscale can take (prevents divide by zero errors).
        :type eps: float, optional

        :var torch.Tensor period_length: The period length parameter. Size/shape of parameter depends on the
            ard_num_dims and batch_shape arguments.
        """

        has_lengthscale = True

        # code from the already-implemented Periodic Kernel
        def _nonkeops_forward(self, x1, x2, diag=False, **kwargs):
            x1_ = x1.div(self.period_length / math.pi)
            x2_ = x2.div(self.period_length / math.pi)

            # We are automatically overriding last_dim_is_batch here so that we can manually sum over dimensions.
            diff = self.covar_dist(x1_, x2_, diag=diag, last_dim_is_batch=True)

            if diag:
                lengthscale = self.lengthscale[..., 0, :, None]
            else:
                lengthscale = self.lengthscale[..., 0, :, None, None]

            exp_term = diff.sin().pow(2.0).div(lengthscale).mul(-2.0)
            exp_term = exp_term.sum(dim=(-2 if diag else -3))

            return exp_term.exp()

        def _keops_forward(self, x1, x2, **kwargs):
            x1_ = x1.div(self.period_length / math.pi)
            x2_ = x2.div(self.period_length / math.pi)
            # return KernelLinearOperator inst only when calculating the whole covariance matrix
            # pass any parameters which are used inside _covar_func as *args to get gradients computed for them
            return KernelLinearOperator(x1_, x2_, lengthscale=self.lengthscale, covar_func=_covar_func, **kwargs)

except ImportError:

    class PeriodicKernel(GPeriodicKernel):
        pass
