from abc import abstractmethod

import torch

from ..kernel import Kernel

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class KeOpsKernel(Kernel):
        @abstractmethod
        def covar_func(self, x1: torch.Tensor, x2: torch.Tensor) -> KEOLazyTensor:
            # We only should use KeOps on big kernel matrices
            # If we would otherwise be performing Cholesky inference, (or when just computing a kernel matrix diag)
            # then don't apply KeOps
            raise NotImplementedError("KeOpsKernels must define a covar_func method")

        def __call__(self, *args, **kwargs):
            # Hotfix for zero gradients. See https://github.com/cornellius-gp/gpytorch/issues/1543
            args = [arg.contiguous() if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.contiguous() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return super().__call__(*args, **kwargs)

except ImportError:

    class KeOpsKernel(Kernel):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("You must have KeOps installed to use a KeOpsKernel")
