#!/usr/bin/env python3

from __future__ import annotations

from linear_operator.operators import KernelLinearOperator
from torch import Tensor

from ..rbf_kernel import RBFKernel as GRBFKernel
from .keops_kernel import _Anysor, _lazify_and_expand_inputs, KeOpsKernel


def _covar_func(x1: _Anysor, x2: _Anysor, **kwargs) -> _Anysor:
    x1_, x2_ = _lazify_and_expand_inputs(x1, x2)
    K = (-((x1_ - x2_) ** 2).sum(-1) / 2).exp()
    return K


class RBFKernel(KeOpsKernel, GRBFKernel):
    r"""
    Implements the RBF kernel using KeOps as a driver for kernel matrix multiplies.

    This class can be used as a drop in replacement for :class:`gpytorch.kernels.RBFKernel` in most cases,
    and supports the same arguments.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
    """

    has_lengthscale = True

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> KernelLinearOperator | Tensor:
        if diag or kwargs.get("last_dim_is_batch", False):
            return GRBFKernel.forward(self, x1, x2, diag=diag, **kwargs)
        kwargs.pop("last_dim_is_batch", None)
        x1_ = x1 / self.lengthscale
        x2_ = x2 / self.lengthscale
        return KernelLinearOperator(x1_, x2_, covar_func=_covar_func, **kwargs)
