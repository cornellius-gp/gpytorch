#!/usr/bin/env python3

from __future__ import annotations

import functools
import warnings
from unittest.mock import MagicMock

import torch

# TODO: Use bool instead of uint8 dtype once pytorch #21113 is in stable release
if isinstance(torch, MagicMock):
    bool_compat = torch.uint8
else:
    bool_compat = (torch.ones(1) > 0).dtype


class DeprecationError(Exception):
    pass


def _deprecated_function_for(old_function_name, function):
    @functools.wraps(function)
    def _deprecated_function(*args, **kwargs):
        warnings.warn(
            f"The `{old_function_name}` function is deprecated. Use `{function.__name__}` instead",
            DeprecationWarning,
        )
        return function(*args, **kwargs)

    return _deprecated_function


def _deprecate_kwarg(kwargs, old_kw, new_kw, new_kw_value):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn(f"The `{old_kw}` argument is deprecated. Use `{new_kw}` instead.", DeprecationWarning)
        if new_kw_value is not None:
            raise ValueError(f"Cannot set both `{old_kw}` and `{new_kw}`")
        return old_kwarg
    return new_kw_value


def _deprecate_kwarg_with_transform(kwargs, old_kw, new_kw, new_kw_value, transform):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn(f"The `{old_kw}` argument is deprecated. Use `{new_kw}` instead.", DeprecationWarning)
        return transform(old_kwarg)
    return new_kw_value


def _deprecated_renamed_method(cls, old_method_name, new_method_name):
    def _deprecated_method(self, *args, **kwargs):
        warnings.warn(
            f"The `{old_method_name}` method is deprecated. Use `{new_method_name}` instead",
            DeprecationWarning,
        )
        return getattr(self, new_method_name)(*args, **kwargs)

    _deprecated_method.__name__ = old_method_name
    setattr(cls, old_method_name, _deprecated_method)
    return cls


def _deprecate_renamed_methods(cls, **renamed_methods):
    for old_method_name, new_method_name in renamed_methods.items():
        _deprecated_renamed_method(cls, old_method_name, new_method_name)
    return cls
