#!/usr/bin/env python3

from torch.autograd import Function

from .. import settings


class Matmul(Function):
    @staticmethod
    def forward(ctx, representation_tree, rhs, *matrix_args):
        ctx.representation_tree = representation_tree
        orig_rhs = rhs

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        lazy_tsr = ctx.representation_tree(*matrix_args)
        res = lazy_tsr._matmul(rhs)

        to_save = [orig_rhs] + list(matrix_args)
        ctx.save_for_backward(*to_save)
        if settings.memory_efficient.off():
            ctx._lazy_tsr = lazy_tsr

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        rhs = ctx.saved_tensors[0]
        matrix_args = ctx.saved_tensors[1:]
        rhs_shape = rhs.shape

        rhs_grad = None
        arg_grads = [None] * len(matrix_args)

        # input_1 gradient
        if any(ctx.needs_input_grad[2:]):
            rhs = rhs.unsqueeze(-1) if (rhs.ndimension() == 1) else rhs
            grad_output_matrix = grad_output.unsqueeze(-1) if grad_output.ndimension() == 1 else grad_output
            arg_grads = ctx.representation_tree(*matrix_args)._quad_form_derivative(grad_output_matrix, rhs)

        # input_2 gradient
        if ctx.needs_input_grad[1]:
            if hasattr(ctx, "_lazy_tsr"):
                lazy_tsr = ctx._lazy_tsr
            else:
                lazy_tsr = ctx.representation_tree(*matrix_args)

            if grad_output.dim() == 1:
                # Confusing Cublas_Sgemv bug when grad_output is single dimensional on GPU.
                rhs_grad = lazy_tsr._t_matmul(grad_output.unsqueeze(-1)).squeeze(-1)
            else:
                rhs_grad = lazy_tsr._t_matmul(grad_output)

            # For broadcasting
            if rhs_grad.dim() > len(rhs_shape):
                rhs_grad = rhs_grad.reshape(-1, *rhs_shape).sum(0)

        return tuple([None] + [rhs_grad] + list(arg_grads))
