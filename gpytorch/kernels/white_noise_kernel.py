#!/usr/bin/env python3

import torch

from . import Kernel
from ..lazy import DiagLazyTensor, ZeroLazyTensor


class WhiteNoiseKernel(Kernel):
    """
    A "random" kernel that adds pre-specified white noise variance to training inputs.
    This is most commonly used in conjunction with another kernel.

    .. note::

        The white noise is only applied to the portion of the kernel matrix that
        represents the training data.

    Args:
        :attr:`variances` (Tensor `n x 1` or `b x n x 1`):
            The random variances to be applied to training inputs.
            `b` and `n` should correspond to the size of the training data.

    Example:
        >>> train_x = torch.randn(10, 5)
        >>> wn_variances = torch.randn(10)
        >>>
        >>> covar_module = gpytorch.kernels.ScaleKernel(
        >>>     gpytorch.kernels.WhiteNoiseKernel(wn_variances) + gpytorch.kernels.MaternKernel(nu=0.5)
        >>> )
        >>> covar = covar_module(train_x)  # Output: LazyVariable of size (10 x 10) (Matern kernel + random variances)
    """

    def __init__(self, variances):
        super(WhiteNoiseKernel, self).__init__()
        self.register_buffer("variances", variances)

    def forward(self, x1, x2, **params):
        if self.training and torch.equal(x1, x2):
            # Reshape into a batch of batch_size diagonal matrices, each of which is
            # (data_size * task_size) x (data_size * task_size)
            return DiagLazyTensor(self.variances.view(*x1.shape[:-2], -1))
        elif x1.size(-2) == x2.size(-2) and x1.size(-2) == self.variances.size(-1) and torch.equal(x1, x2):
            return DiagLazyTensor(self.variances.view(*x1.shape[:-2], -1))
        else:
            return ZeroLazyTensor(*x1.shape[:-2], x1.size(-2), x2.size(-2), dtype=x1.dtype, device=x1.device)
