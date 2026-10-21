from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class RBFCovariance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x1: torch.Tensor,
        x2: torch.Tensor,
        lengthscale: torch.Tensor,
        sq_dist_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("RBFCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("RBFCovariance cannot handle multiple lengthscales")
        needs_grad = any(ctx.needs_input_grad)
        x1_ = x1.div(lengthscale)
        x2_ = x2.div(lengthscale)
        unitless_sq_dist = sq_dist_func(x1_, x2_)
        # clone because inplace operations will mess with what's saved for backward
        unitless_sq_dist_ = unitless_sq_dist.clone() if needs_grad else unitless_sq_dist
        covar_mat = unitless_sq_dist_.div_(-2.0).exp_()
        if needs_grad:
            d_output_d_input = unitless_sq_dist.mul_(covar_mat).div_(lengthscale)
            ctx.save_for_backward(x1, x2, lengthscale, d_output_d_input)
            ctx.sq_dist_func = sq_dist_func
        return covar_mat

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, None, torch.Tensor, None]:
        x1, x2, lengthscale, d_output_d_input = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Forward's cached derivative was computed without recording its
            # graph. Recompute only for higher derivatives, retaining the
            # lengthscale dependence of both distances and covariance values.
            unitless_sq_dist = ctx.sq_dist_func(x1 / lengthscale, x2 / lengthscale)
            d_output_d_input = unitless_sq_dist * (-0.5 * unitless_sq_dist).exp() / lengthscale
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None
