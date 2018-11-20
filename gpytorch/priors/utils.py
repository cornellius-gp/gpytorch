#!/usr/bin/env python3


def _bufferize_attributes(module, attributes):
    attr_clones = {attr: getattr(module, attr).clone() for attr in attributes}
    for attr, value in attr_clones.items():
        delattr(module, attr)
        module.register_buffer(attr, value)


def _del_attributes(module, attributes, raise_on_error=False):
    for attr in attributes:
        try:
            delattr(module, attr)
        except AttributeError as e:
            if raise_on_error:
                raise e
    return module
