#!/usr/bin/env python3

import warnings
from typing import Any, Union

import torch
from linear_operator import operators

from linear_operator.utils import errors


class CachingError(RuntimeError):
    pass


def raise_caching_error_if_overwriting_accidentally(
    cached_quantity: Union[torch.Tensor, operators.LinearOperator], overwrite: bool
):
    if not overwrite:
        if cached_quantity is not None:
            raise CachingError(
                "Trying to fill a cache which has not been cleared without explicitly "
                "choosing to `overwrite`."
                "Ensure you are clearing caches appropriately or you want to overwrite."
            )


def __getattr__(name: str) -> Any:
    if hasattr(errors, name):
        warnings.warn(
            f"gpytorch.utils.errors.{name} has been deprecated. Use linear_operator.utils.error.{name} instead.",
            DeprecationWarning,
        )
        return getattr(errors, name)
    raise AttributeError(f"module gpytorch.utils.errors has no attribute {name}")
