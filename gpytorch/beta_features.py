#!/usr/bin/env python3

import warnings
from .settings import _feature_flag
from .settings import fast_pred_var as _fast_pred_var
from .settings import fast_pred_samples as _fast_pred_samples


class _moved_beta_feature(object):
    def __init__(self, new_cls, orig_name=None):
        self.new_cls = new_cls
        self.orig_name = orig_name if orig_name is not None else f"gpytorch.settings.{new_cls.__name__}"

    def __call__(self, *args, **kwargs):
        warnings.warn(
            f"`{self.orig_name}` has moved to `gpytorch.settings.{self.new_cls.__name__}`.",
            DeprecationWarning
        )
        return self.new_cls(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.new_cls, name)


fast_pred_var = _moved_beta_feature(_fast_pred_var)
fast_pred_samples = _moved_beta_feature(_fast_pred_samples)


class diagonal_correction(_feature_flag):
    """
    Add a diagonal correction to scalable inducing point methods
    """

    _state = True


class default_preconditioner(_feature_flag):
    """
    Add a diagonal correction to scalable inducing point methods
    """

    pass


__all__ = ["fast_pred_var", "fast_pred_samples", "diagonal_correction", "default_preconditioner"]
