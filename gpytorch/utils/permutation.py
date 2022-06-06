#!/usr/bin/env python3
# DEPRECATED_UTIL

from __future__ import annotations

import warnings
from typing import Any

import linear_operator


def __getattr__(name: str) -> Any:
    if hasattr(linear_operator.utils.permutation, name):
        warnings.warn(
            f"gpytorch.utils.permutation.{name} is deprecated. Use linear_operator.utils.permutation.{name} instead.",
            DeprecationWarning,
        )
        return getattr(linear_operator.utils.permutation, name)
    raise AttributeError(f"module gpytorch.utils.permutation has no attribute {name}")
