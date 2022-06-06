#!/usr/bin/env python3
# DEPRECATED_UTIL

from __future__ import annotations

import warnings
from typing import Any

import linear_operator


def __getattr__(name: str) -> Any:
    if hasattr(linear_operator.utils.sparse, name):
        warnings.warn(
            f"gpytorch.utils.sparse.{name} is deprecated. Use linear_operator.utils.sparse.{name} instead.",
            DeprecationWarning,
        )
        return getattr(linear_operator.utils.sparse, name)
    raise AttributeError(f"module gpytorch.utils.sparse has no attribute {name}")
