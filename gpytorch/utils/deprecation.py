#!/usr/bin/env python3

import warnings


class DeprecationError(Exception):
    pass


def _deprecate_kwarg(kwargs, old_kw, new_kw, new_kw_value):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn("The `{}` argument is deprecated. Use `{}` instead.".format(old_kw, new_kw), DeprecationWarning)
        if new_kw_value is not None:
            raise ValueError("Cannot set both `{}` and `{}`".format(old_kw, new_kw))
        return old_kwarg
    return new_kw_value
