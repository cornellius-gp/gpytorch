#!/usr/bin/env python3

import warnings
from typing import Any

from linear_operator.utils import errors


class CachingError(RuntimeError):
    pass


def __getattr__(name: str) -> Any:
    if hasattr(errors, name):
        warnings.warn(
            f"gpytorch.utils.errors.{name} has been deprecated. Use linear_operator.utils.error.{name} instead.",
            DeprecationWarning,
        )
        return getattr(errors, name)
    raise AttributeError(f"module gpytorch.utils.errors has no attribute {name}")
