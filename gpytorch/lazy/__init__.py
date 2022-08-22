#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

import torch
from linear_operator import LinearOperator, operators
from linear_operator.operators.cat_linear_operator import cat as _cat

from .lazy_evaluated_kernel_tensor import LazyEvaluatedKernelTensor
from .lazy_tensor import delazify, deprecated_lazy_tensor
from .non_lazy_tensor import lazify

# We will dynamically import LazyTensor/NonLazyTensor to trigger DeprecationWarnings


_deprecated_lazy_tensors = {
    "AddedDiagLazyTensor": deprecated_lazy_tensor(operators.AddedDiagLinearOperator),
    "BatchRepeatLazyTensor": deprecated_lazy_tensor(operators.BatchRepeatLinearOperator),
    "BlockDiagLazyTensor": deprecated_lazy_tensor(operators.BlockDiagLinearOperator),
    "BlockInterleavedLazyTensor": deprecated_lazy_tensor(operators.BlockInterleavedLinearOperator),
    "BlockLazyTensor": deprecated_lazy_tensor(operators.BlockLinearOperator),
    "CatLazyTensor": deprecated_lazy_tensor(operators.CatLinearOperator),
    "CholLazyTensor": deprecated_lazy_tensor(operators.CholLinearOperator),
    "ConstantMulLazyTensor": deprecated_lazy_tensor(operators.ConstantMulLinearOperator),
    "ConstantDiagLazyTensor": deprecated_lazy_tensor(operators.ConstantDiagLinearOperator),
    "DiagLazyTensor": deprecated_lazy_tensor(operators.DiagLinearOperator),
    "IdentityLazyTensor": deprecated_lazy_tensor(operators.IdentityLinearOperator),
    "InterpolatedLazyTensor": deprecated_lazy_tensor(operators.InterpolatedLinearOperator),
    "KeOpsLazyTensor": deprecated_lazy_tensor(operators.KeOpsLinearOperator),
    "KroneckerProductAddedDiagLazyTensor": deprecated_lazy_tensor(operators.KroneckerProductAddedDiagLinearOperator),
    "KroneckerProductDiagLazyTensor": deprecated_lazy_tensor(operators.KroneckerProductDiagLinearOperator),
    "KroneckerProductLazyTensor": deprecated_lazy_tensor(operators.KroneckerProductLinearOperator),
    "KroneckerProductTriangularLazyTensor": deprecated_lazy_tensor(operators.KroneckerProductTriangularLinearOperator),
    "LowRankRootAddedDiagLazyTensor": deprecated_lazy_tensor(operators.LowRankRootAddedDiagLinearOperator),
    "LowRankRootLazyTensor": deprecated_lazy_tensor(operators.LowRankRootLinearOperator),
    "MatmulLazyTensor": deprecated_lazy_tensor(operators.MatmulLinearOperator),
    "MulLazyTensor": deprecated_lazy_tensor(operators.MulLinearOperator),
    "PsdSumLazyTensor": deprecated_lazy_tensor(operators.PsdSumLinearOperator),
    "RootLazyTensor": deprecated_lazy_tensor(operators.RootLinearOperator),
    "SumBatchLazyTensor": deprecated_lazy_tensor(operators.SumBatchLinearOperator),
    "SumKroneckerLazyTensor": deprecated_lazy_tensor(operators.SumKroneckerLinearOperator),
    "SumLazyTensor": deprecated_lazy_tensor(operators.SumLinearOperator),
    "ToeplitzLazyTensor": deprecated_lazy_tensor(operators.ToeplitzLinearOperator),
    "TriangularLazyTensor": deprecated_lazy_tensor(operators.TriangularLinearOperator),
    "ZeroLazyTensor": deprecated_lazy_tensor(operators.ZeroLinearOperator),
}


def cat(
    inputs: Tuple[Union[LinearOperator, torch.Tensor], ...], dim: int = 0, output_device: Optional[torch.device] = None
) -> Union[torch.Tensor, LinearOperator]:
    warnings.warn("gpytorch.lazy.cat is deprecated in favor of linear_operator.cat")
    return _cat(inputs, dim=dim, output_device=output_device)


__all__ = [
    "delazify",
    "lazify",
    "cat",
    "LazyEvaluatedKernelTensor",
    "LazyTensor",
]


def __getattr__(name: str) -> Any:
    warnings.warn(
        "GPyTorch will be replacing all LazyTensor functionality with the linear operator package. "
        "Replace all references to gpytorch.lazy.*LazyTensor with linear_operator.operators.*LinearOperator.",
        DeprecationWarning,
    )
    if name == "LazyTensor":
        from .lazy_tensor import LazyTensor

        return deprecated_lazy_tensor(LazyTensor)
    elif name == "NonLazyTensor":
        from .non_lazy_tensor import NonLazyTensor

        return deprecated_lazy_tensor(NonLazyTensor)
    elif name in _deprecated_lazy_tensors:
        return _deprecated_lazy_tensors[name]
    raise AttributeError(f"module gpytorch.lazy has no attribute {name}")
