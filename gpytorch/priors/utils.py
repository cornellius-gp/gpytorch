#!/usr/bin/env python3

from torch.distributions import TransformedDistribution


def _bufferize_attributes(module, attributes):
    r"""
    Adds the parameters of the prior as a torch buffer to enable saving/
    loading to/from state_dicts.
    For TransformedDistributions adds a _buffered_ attribute to the
    parameters. This enables its parameters to be saved and
    loaded to/from state_dicts, as the original parameters cannot be.
    """
    if isinstance(module, TransformedDistribution):
        for attr in attributes:
            module.register_buffer(f"_buffered_{attr}", getattr(module, attr))
    else:
        attr_clones = {attr: getattr(module, attr).clone() for attr in attributes}
        for attr, value in attr_clones.items():
            delattr(module, attr)
            module.register_buffer(attr, value)


def _load_buffered_to_base_dist(module):
    r"""Loads the _buffered_ attributes to the parameters of a torch
    TransformedDistribution. This enables its parameters to be saved and
    loaded to/from state_dicts, as the original parameters cannot be.
    """
    buffered_str = "_buffered_"
    buffered_attrs = [attr for attr in dir(module) if buffered_str in attr]
    for buffered_attr in buffered_attrs:
        base_attr_name = buffered_attr.replace(buffered_str, "")
        setattr(module.base_dist, base_attr_name, getattr(module, buffered_attr))


def _del_attributes(module, attributes, raise_on_error=False):
    for attr in attributes:
        try:
            delattr(module, attr)
        except AttributeError as e:
            if raise_on_error:
                raise e
    return module
