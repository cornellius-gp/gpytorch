#!/usr/bin/env python3

import warnings
from typing import Union

import torch
from linear_operator import LinearOperator, operators, to_linear_operator

from .lazy_tensor import deprecated_lazy_tensor


def lazify(obj: Union[LinearOperator, torch.Tensor]) -> operators.DenseLinearOperator:
    warnings.warn("gpytorch.lazy.lazify is deprecated in favor of linear_operator.to_linear_operator")
    return to_linear_operator(obj)


NonLazyTensor = deprecated_lazy_tensor(operators.DenseLinearOperator)
