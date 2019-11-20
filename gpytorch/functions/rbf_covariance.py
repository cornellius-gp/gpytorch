import torch


class RBFCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, sq_dist_func):
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
            ctx.save_for_backward(d_output_d_input)
        return covar_mat

    @staticmethod
    def backward(ctx, grad_output):
        d_output_d_input = ctx.saved_tensors[0]
        lengthscale_grad = grad_output * d_output_d_input
        return None, None, lengthscale_grad, None
