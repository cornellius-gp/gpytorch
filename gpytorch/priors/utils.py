#!/usr/bin/env python3

from __future__ import annotations

from torch.distributions import TransformedDistribution


# Prefix for buffered attributes in TransformedDistributions.
# These are copies of the base distribution attributes, enabling state_dict
# save/load since the original attributes are properties and cannot be bufferized.
BUFFERED_PREFIX = "_buffered_"


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
            module.register_buffer(f"{BUFFERED_PREFIX}{attr}", getattr(module, attr))
    else:
        attr_clones = {attr: getattr(module, attr).clone() for attr in attributes}
        for attr, value in attr_clones.items():
            delattr(module, attr)
            module.register_buffer(attr, value)


def _load_transformed_to_base_dist(module):
    r"""loads the _buffered_ attributes to the parameters of a torch
    TransformedDistribution. This enables its parameters to be saved and
    loaded to/from state_dicts, as the original attributes cannot be.
    """
    for attr in dir(module):
        if BUFFERED_PREFIX in attr:
            base_attr_name = attr.replace(BUFFERED_PREFIX, "")
            setattr(module.base_dist, base_attr_name, getattr(module, attr))


def _del_attributes(module, attributes, raise_on_error=False):
    for attr in attributes:
        try:
            delattr(module, attr)
        except AttributeError as e:
            if raise_on_error:
                raise e
    return module
