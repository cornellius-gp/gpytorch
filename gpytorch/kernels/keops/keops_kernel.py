from abc import abstractmethod
from typing import Any

import torch
from torch import Tensor

from ... import settings
from ..kernel import Kernel

try:
    import pykeops  # noqa F401

    class KeOpsKernel(Kernel):
        @abstractmethod
        def _nonkeops_forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs: Any):
            r"""
            Computes the covariance matrix (or diagonal) without using KeOps.
            This function must implement both the diag=True and diag=False options.
            """
            raise NotImplementedError

        @abstractmethod
        def _keops_forward(self, x1: Tensor, x2: Tensor, **kwargs: Any):
            r"""
            Computes the covariance matrix with KeOps.
            This function only implements the diag=False option, and no diag keyword should be passed in.
            """
            raise NotImplementedError

        def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs: Any):
            if diag:
                return self._nonkeops_forward(x1, x2, diag=True, **kwargs)
            elif x1.size(-2) < settings.max_cholesky_size.value() or x2.size(-2) < settings.max_cholesky_size.value():
                return self._nonkeops_forward(x1, x2, diag=False, **kwargs)
            else:
                return self._keops_forward(x1, x2, **kwargs)

        def __call__(self, *args: Any, **kwargs: Any):
            # Hotfix for zero gradients. See https://github.com/cornellius-gp/gpytorch/issues/1543
            args = [arg.contiguous() if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.contiguous() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return super().__call__(*args, **kwargs)

except ImportError:

    class KeOpsKernel(Kernel):
        def __init__(self, *args: Any, **kwargs: Any):
            raise RuntimeError("You must have KeOps installed to use a KeOpsKernel")
