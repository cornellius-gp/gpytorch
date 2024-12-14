#!/usr/bin/env python3
import math

from linear_operator.operators import KernelLinearOperator
from torch import Tensor

from .keops_kernel import _Anysor, _lazify_and_expand_inputs, KeOpsKernel


def _covar_func(x1: _Anysor, x2: _Anysor, nu: float = 2.5, **params) -> _Anysor:
    x1_, x2_ = _lazify_and_expand_inputs(x1, x2)

    sq_distance = ((x1_ - x2_) ** 2).sum(-1)
    distance = (sq_distance + 1e-20).sqrt()
    # ^^ Need to add epsilon to prevent small negative values with the sqrt
    # backward pass (otherwise we get NaNs).
    # using .clamp(1e-20, math.inf) doesn't work in KeOps; it also creates NaNs
    exp_component = (-math.sqrt(nu * 2) * distance).exp()

    if nu == 0.5:
        constant_component = 1
    elif nu == 1.5:
        constant_component = (math.sqrt(3) * distance) + 1
    elif nu == 2.5:
        constant_component = (math.sqrt(5) * distance) + (1 + 5.0 / 3.0 * sq_distance)

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

    def __init__(self, nu: float = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.nu = nu

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> KernelLinearOperator:
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean) / self.lengthscale
        x2_ = (x2 - mean) / self.lengthscale
        # return KernelLinearOperator inst only when calculating the whole covariance matrix
        res = KernelLinearOperator(x1_, x2_, covar_func=_covar_func, nu=self.nu, **kwargs)

        # TODO: diag=True mode will be removed with the GpyTorch 2.0 PR to remove LazyEvaluatedKernelTensor
        # (it will be replaced by a `_symmetric_diag` method for quickly computing the diagonals of symmetric matrices)
        if diag:
            return res.diagonal(dim1=-1, dim2=-2)
        else:
            return res
