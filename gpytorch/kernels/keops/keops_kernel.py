import torch
from pykeops.torch import LazyTensor as KEOLazyTensor
from ..kernel import Kernel
from abc import abstractmethod


class KeOpsKernel(Kernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def covar_func(self, x1: torch.Tensor, x2: torch.Tensor) -> KEOLazyTensor:
        raise NotImplementedError("KeOpsKernels must define a covar_func method")
