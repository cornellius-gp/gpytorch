import warnings
from typing import Any, Tuple, Union

import torch
from linear_operator import LinearOperator
from torch import Tensor

from ... import settings
from ..kernel import Kernel

try:
    import pykeops  # noqa F401
    from pykeops.torch import LazyTensor

    _Anysor = Union[Tensor, LazyTensor]

    def _lazify_and_expand_inputs(
        x1: Tensor, x2: Tensor
    ) -> Tuple[Union[Tensor, LazyTensor], Union[Tensor, LazyTensor]]:
        r"""
        Potentially wrap inputs x1 and x2 as KeOps LazyTensors,
        depending on whether or not we want to use KeOps under the hood or not.
        """
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        if _use_keops(x1, x2):
            res = LazyTensor(x1_), LazyTensor(x2_)
            return res
        return x1_, x2_

    def _use_keops(x1: Tensor, x2: Tensor) -> bool:
        r"""
        Determine whether or not to use KeOps under the hood
        This largely depends on the size of the kernel matrix

        There are situations where we do not want the KeOps linear operator to use KeOps under the hood.
        See https://github.com/cornellius-gp/gpytorch/pull/1319
        """
        return (
            settings.use_keops.on()
            and x1.size(-2) >= settings.max_cholesky_size.value()
            and x2.size(-2) >= settings.max_cholesky_size.value()
        )

    class KeOpsKernel(Kernel):
        def __call__(self, *args: Any, **kwargs: Any) -> Union[LinearOperator, Tensor, LazyTensor]:
            # Hotfix for zero gradients. See https://github.com/cornellius-gp/gpytorch/issues/1543
            args = [arg.contiguous() if torch.is_tensor(arg) else arg for arg in args]
            kwargs = {k: v.contiguous() if torch.is_tensor(v) else v for k, v in kwargs.items()}
            return super().__call__(*args, **kwargs)

except ImportError:

    _Anysor = Tensor

    def _lazify_and_expand_inputs(x1: Tensor, x2: Tensor) -> Tuple[Tensor, Tensor]:
        x1_ = x1[..., :, None, :]
        x2_ = x2[..., None, :, :]
        return x1_, x2_

    def _use_keops(x1: Tensor, x2: Tensor) -> bool:
        return False

    class KeOpsKernel(Kernel):
        def __call__(self, *args: Any, **kwargs: Any) -> Union[LinearOperator, Tensor]:
            warnings.warn(
                "KeOps is not installed. " f"{type(self)} will revert to the the non-keops version of this kernel.",
                RuntimeWarning,
            )
            return super().__call__(*args, **kwargs)
