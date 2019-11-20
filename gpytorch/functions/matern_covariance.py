import math

import torch


class MaternCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, nu, dist_func):
        if any(ctx.needs_input_grad[:2]):
            raise RuntimeError("MaternCovariance cannot compute gradients with " "respect to x1 and x2")
        if lengthscale.size(-1) > 1:
            raise ValueError("MaternCovariance cannot handle multiple lengthscales")
        # Subtract mean for numerical stability. Won't affect computations
        # because covariance matrix is stationary.
        needs_grad = any(ctx.needs_input_grad)
        mean = x1.reshape(-1, x1.size(-1)).mean(0)[(None,) * (x1.dim() - 1)]
        x1_ = (x1 - mean).div(lengthscale)
        x2_ = (x2 - mean).div(lengthscale)
        scaled_unitless_dist = dist_func(x1_, x2_).mul_(math.sqrt(2 * nu))
        if nu == 0.5:
            # 1 kernel sized Tensor if no grad else 2
            scaled_unitless_dist_ = scaled_unitless_dist.clone() if needs_grad else scaled_unitless_dist
            exp_component = scaled_unitless_dist_.neg_().exp_()
            covar_mat = exp_component
            if needs_grad:
                d_output_d_input = scaled_unitless_dist.div_(lengthscale).mul_(exp_component)
        elif nu == 1.5:
            # 2 kernel sized Tensors if no grad else 3
            if needs_grad:
                scaled_unitless_dist_ = scaled_unitless_dist.clone()
            linear_term = scaled_unitless_dist.clone().add_(1)
            exp_component = scaled_unitless_dist.neg_().exp_()
            covar_mat = linear_term.mul_(exp_component)
            if needs_grad:
                d_output_d_input = scaled_unitless_dist_.pow_(2).div_(lengthscale).mul_(exp_component)
        elif nu == 2.5:
            # 3 kernel sized Tensors if no grad else 4
            linear_term = scaled_unitless_dist.clone().add_(1)
            quadratic_term = scaled_unitless_dist.clone().pow_(2).div_(3)
            exp_component = scaled_unitless_dist.neg_().exp_()
            if needs_grad:
                covar_mat = (linear_term + quadratic_term).mul_(exp_component)
                d_output_d_input = linear_term.mul_(quadratic_term).mul_(exp_component).div_(lengthscale)
            else:
                covar_mat = exp_component.mul_(linear_term.add_(quadratic_term))
        if needs_grad:
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None, None
