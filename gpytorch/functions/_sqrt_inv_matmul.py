#!/usr/bin/env python3

import torch
from torch.autograd import Function
from ..utils.contour_integral_quad import sqrt_matmul

class SqrtInvMatmul(Function):
    @staticmethod
    def forward(ctx, representation_tree, *args):
        ctx.representation_tree = representation_tree
        rhs, *matrix_args = args
        ctx.lazy_tsr = ctx.representation_tree(*matrix_args)

        print("stuff")
        res = sqrt_matmul(ctx.lazy_tsr, rhs, inverse=True, num_quad_samples=10)
        ctx.save_for_backward(res, rhs, *matrix_args)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        rhs, product, *matrix_args = ctx.saved_tensors

        rhs_grad = None
        matrix_arg_grads = ctx.lazy_tsr._quad_form_derivative(
            sqrt_matmul(ctx.lazy_tsr, product, inverse=True).mul(-1),
            sqrt_matmul(ctx.lazy_tsr, grad_output, inverse=True)
        )
        return tuple([None] + [rhs_grad] + list(matrix_arg_grads))
