#!/usr/bin/env python3

import torch


def _mul_broadcast_shape(shape_a, shape_b):
    """Compute dimension of mul operation on shapes (supports broadcasting)"""
    bc_shape = []
    len_a, len_b = len(shape_a), len(shape_b)
    for i in range(1, 1 + min(len_a, len_b)):
        s_a, s_b = shape_a[-i], shape_b[-i]
        if s_a != s_b:
            if min(s_a, s_b) > 1:
                raise RuntimeError("batch sizes not broadcastable")
            bc_shape.insert(0, max(s_a, s_b))
        else:
            bc_shape.insert(0, s_a)

    # fill remaining dimensions on the left if necessary
    delta_len = len_a - len_b
    if delta_len > 0:
        bc_shape = list(shape_a[:delta_len]) + bc_shape
    elif delta_len < 0:
        bc_shape = list(shape_b[:-delta_len]) + bc_shape

    return torch.Size(bc_shape)


def _matmul_broadcast_shape(shape_a, shape_b):
    """Compute dimension of matmul operation on shapes (supports broadcasting)"""
    m, n, p = shape_a[-2], shape_a[-1], shape_b[-1]

    if len(shape_b) == 1:
        if n != p:
            raise RuntimeError("Incompatible dimensions for matmul")
        return shape_a[:-1]

    if n != shape_b[-2]:
        raise RuntimeError("Incompatible dimensions for matmul")

    tail_shape = torch.Size([m, p])
    bc_shape = _mul_broadcast_shape(shape_a[:-2], shape_b[:-2])
    return bc_shape + tail_shape
