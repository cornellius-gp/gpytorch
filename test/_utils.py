#!/usr/bin/env python3

import torch


def approx_equal(self, other, epsilon=1e-4):
    """
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
    """
    if self.size() != other.size():
        raise RuntimeError(
            "Size mismatch between self ({self}) and other ({other})".format(self=self.size(), other=other.size())
        )
    return torch.max((self - other).abs()) <= epsilon
