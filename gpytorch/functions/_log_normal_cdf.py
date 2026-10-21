#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx


class LogNormalCDF(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, z: Tensor) -> Tensor:
        ctx.save_for_backward(z)
        return torch.special.log_ndtr(z)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tensor:
        (z,) = ctx.saved_tensors
        # Recompute from z rather than a cached forward approximation so that
        # differentiating backward retains the derivative of the Mills ratio.
        log_derivative = -0.5 * z.square() - torch.special.log_ndtr(z) - 0.5 * math.log(2 * math.pi)
        return grad_output * log_derivative.exp()
