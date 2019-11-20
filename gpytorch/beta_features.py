#!/usr/bin/env python3

import warnings

from .settings import _feature_flag, _value_context


class _moved_beta_feature(object):
    def __init__(self, new_cls, orig_name=None):
        self.new_cls = new_cls
        self.orig_name = orig_name if orig_name is not None else "gpytorch.settings.{}".format(new_cls.__name__)

    def __call__(self, *args, **kwargs):
        warnings.warn(
            "`{}` has moved to `gpytorch.settings.{}`.".format(self.orig_name, self.new_cls.__name__),
            DeprecationWarning,
        )
        return self.new_cls(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.new_cls, name)


class checkpoint_kernel(_value_context):
    """
    Should the kernel be computed in chunks with checkpointing or not? (Default, no)

    If `split_size = 0`:
        The kernel is computed explicitly. During training, the kernel matrix is
        kept in memory for the backward pass. This is the fastest option but the
        most memory intensive.
    If `split_size > 0`:
        The kernel is never fully computed or stored. Instead, the kernel is only
        accessed through matrix multiplication. The matrix multiplication is
        computed in `segments` chunks. This is slower, but requires significantly less memory.

    Default: 0
    """

    _global_value = 0


class default_preconditioner(_feature_flag):
    """
    Add a diagonal correction to scalable inducing point methods
    """

    pass


__all__ = ["checkpoint_kernel", "default_preconditioner"]
