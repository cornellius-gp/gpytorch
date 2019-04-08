#!/usr/bin/env python3

import warnings
import functools


class DeprecationError(Exception):
    pass


class _ClassWithDeprecatedBatchSize(object):
    def _batch_shape_state_dict_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        try:
            current_state_dict = self.state_dict()
            for name, param in current_state_dict.items():
                load_param = state_dict[prefix + name]
                if load_param.dim() == param.dim() + 1:
                    warnings.warn(
                        f"The supplied state_dict contains a parameter ({prefix + name}) with an extra batch "
                        f"dimension ({load_param.shape} vs {param.shape}).\nDefault batch shapes are now "
                        "deprecated in GPyTorch. You may wish to re-save your model.", DeprecationWarning
                    )
                    load_param.squeeze_(0)
        except Exception:
            pass


def _deprecated_function_for(old_function_name, function):
    @functools.wraps(function)
    def _deprecated_function(*args, **kwargs):
        warnings.warn(
            "The `{}` function is deprecated. Use `{}` instead".format(old_function_name, function.__name__),
            DeprecationWarning
        )
        return function(*args, **kwargs)

    return _deprecated_function


def _deprecate_kwarg(kwargs, old_kw, new_kw, new_kw_value):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn("The `{}` argument is deprecated. Use `{}` instead.".format(old_kw, new_kw), DeprecationWarning)
        if new_kw_value is not None:
            raise ValueError("Cannot set both `{}` and `{}`".format(old_kw, new_kw))
        return old_kwarg
    return new_kw_value


def _deprecate_kwarg_with_transform(kwargs, old_kw, new_kw, new_kw_value, transform):
    old_kwarg = kwargs.get(old_kw)
    if old_kwarg is not None:
        warnings.warn("The `{}` argument is deprecated. Use `{}` instead.".format(old_kw, new_kw), DeprecationWarning)
        return transform(old_kwarg)
    return new_kw_value


def _deprecated_renamed_method(cls, old_method_name, new_method_name):
    def _deprecated_method(self, *args, **kwargs):
        warnings.warn(
            "The `{}` method is deprecated. Use `{}` instead".format(old_method_name, new_method_name),
            DeprecationWarning
        )
        return getattr(self, new_method_name)(*args, **kwargs)

    _deprecated_method.__name__ = old_method_name
    setattr(cls, old_method_name, _deprecated_method)
    return cls


def _deprecate_renamed_methods(cls, **renamed_methods):
    for old_method_name, new_method_name in renamed_methods.items():
        _deprecated_renamed_method(cls, old_method_name, new_method_name)
    return cls
