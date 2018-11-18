from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def _get_inv_param_transform(param_transform, inv_param_transform=None):
    reg_inv_tf = TRANSFORM_REGISTRY.get(param_transform, None)
    if reg_inv_tf is None:
        if inv_param_transform is None:
            raise RuntimeError("Must specify inv_param_transform for custom param_transforms")
        return inv_param_transform
    elif inv_param_transform is not None and reg_inv_tf != inv_param_transform:
        raise RuntimeError("TODO")
    return reg_inv_tf


TRANSFORM_REGISTRY = {torch.exp: torch.log, torch.nn.functional.softplus: inv_softplus}
