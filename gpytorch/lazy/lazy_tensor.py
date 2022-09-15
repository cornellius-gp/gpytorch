#!/usr/bin/env python3

import warnings
from functools import wraps
from typing import Union

import torch
from linear_operator import LinearOperator, to_dense


def _add_deprecated_method(cls: type, old_method_name: str, new_method_name: str):
    method = getattr(cls, new_method_name)

    @wraps(method)
    def _deprecated_method(self, *args, **kwargs):
        warnings.warn(
            f"LazyTensor#{old_method_name} has been renamed to LinearOperator#{method.__name__}", DeprecationWarning
        )
        return method(self, *args, **kwargs)

    setattr(cls, old_method_name, _deprecated_method)


def delazify(obj: Union[LinearOperator, torch.Tensor]) -> torch.Tensor:
    warnings.warn("gpytorch.lazy.delazify is deprecated in favor of linear_operator.to_dense")
    return to_dense(obj)


def deprecated_lazy_tensor(_LinearOperatorClass: type) -> type:
    __orig_init__ = getattr(_LinearOperatorClass, "__init__")

    def __init__(self, *args, **kwargs):
        new_kwargs = dict()
        for name, val in kwargs.items():
            if "lazy_tensor" in name:
                new_name = name.replace("lazy_tensor", "linear_op")
                warnings.warn(
                    f"The kwarg {name} for {self.__class__.__name__}.__init__ is deprecated. Use "
                    f"the kwarg {new_name} instead.",
                    DeprecationWarning,
                )
                new_kwargs[new_name] = val
            else:
                new_kwargs[name] = val

        return __orig_init__(self, *args, **new_kwargs)

    def symeig(self, eigenvectors=True):
        warnings.warn(
            "LazyTensor#symeig has been renamed to LinearOperator#eigh/eigvalsh. "
            "(eigh replaces symeig(eigenvectors=True); eigvalsh replaces symeig(eigenvectors=False).)",
            DeprecationWarning,
        )
        if eigenvectors:
            return self.eigh()
        else:
            return self.eigvalsh(), None

    def __getattr__(self, name):
        if "lazy_tensor" in name:
            new_name = name.replace("lazy_tensor", "linear_op")
            if hasattr(self, new_name):
                warnings.warn(
                    f"The property {self.__class__.__name__}#{name} is depreated. Use "
                    f"{self.__class__.__name__}#{new_name} instead."
                )
                return getattr(self, new_name)
        raise AttributeError("Unknown attribute {name} for {self.__name__.__class__}")

    _add_deprecated_method(_LinearOperatorClass, "_approx_diag", "_approx_diagonal")
    _add_deprecated_method(_LinearOperatorClass, "_quad_form_derivative", "_bilinear_derivative")
    _add_deprecated_method(_LinearOperatorClass, "add_diag", "add_diagonal")
    _add_deprecated_method(_LinearOperatorClass, "diag", "diagonal")
    _add_deprecated_method(_LinearOperatorClass, "evaluate", "to_dense")
    _add_deprecated_method(_LinearOperatorClass, "inv_matmul", "solve")
    setattr(_LinearOperatorClass, "symeig", symeig)
    setattr(_LinearOperatorClass, "__getattr__", __getattr__)
    setattr(_LinearOperatorClass, "__init__", __init__)
    return _LinearOperatorClass


LazyTensor = deprecated_lazy_tensor(LinearOperator)
