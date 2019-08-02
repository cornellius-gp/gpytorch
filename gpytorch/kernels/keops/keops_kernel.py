import torch
from ..kernel import Kernel
from abc import abstractmethod

try:
    from pykeops.torch import LazyTensor as KEOLazyTensor

    class KeOpsKernel(Kernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @abstractmethod
        def covar_func(self, x1: torch.Tensor, x2: torch.Tensor) -> KEOLazyTensor:
            raise NotImplementedError("KeOpsKernels must define a covar_func method")

except ImportError:
    class KeOpsKernel(Kernel):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("You must have KeOps installed to use a KeOpsKernel")
