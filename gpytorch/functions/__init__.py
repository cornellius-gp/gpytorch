#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Any

import linear_operator
import torch

from ._log_normal_cdf import LogNormalCDF
from .matern_covariance import MaternCovariance
from .rbf_covariance import RBFCovariance


def log_normal_cdf(x):
    """
    Computes the element-wise log standard normal CDF of an input tensor x.

    This function should always be preferred over calling normal_cdf and taking the log
    manually, as it is more numerically stable.
    """
    return LogNormalCDF.apply(x)


def logdet(mat):
    warnings.warn("gpytorch.logdet is deprecated. Use torch.logdet instead.", DeprecationWarning)
    return torch.logdet(mat)


def matmul(mat, rhs):
    warnings.warn("gpytorch.matmul is deprecated. Use torch.matmul instead.", DeprecationWarning)
    return torch.matmul(mat, rhs)


def inv_matmul(mat, right_tensor, left_tensor=None):
    warnings.warn("gpytorch.inv_matmul is deprecated. Use gpytorch.solve instead.", DeprecationWarning)
    return linear_operator.solve(right_tensor, left_tensor=None)


__all__ = [
    "MaternCovariance",
    "RBFCovariance",
    "inv_matmul",
    "logdet",
    "log_normal_cdf",
    "matmul",
]


def __getattr__(name: str) -> Any:
    if hasattr(linear_operator.functions, name):
        warnings.warn(
            f"gpytorch.functions.{name} is deprecated. Use linear_operator.functions.{name} instead.",
            DeprecationWarning,
        )
        return getattr(linear_operator.functions, name)
    raise AttributeError(f"module gpytorch.functions has no attribute {name}.")
