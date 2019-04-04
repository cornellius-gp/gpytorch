#!/usr/bin/env python3

import torch


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res
