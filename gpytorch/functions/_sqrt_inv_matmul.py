#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings
from ..utils.contour_integral_quad import contour_integral_quad


class SqrtInvMatmul(Function):
    @staticmethod
    def forward(ctx, representation_tree, *args):
        ctx.representation_tree = representation_tree
        rhs, *matrix_args = args
        ctx.lazy_tsr = ctx.representation_tree(*matrix_args)

        rhs_solves, weights, shifts = contour_integral_quad(
            ctx.lazy_tsr, rhs, inverse=True, num_contour_quadrature=settings.num_contour_quadrature.value()
        )
        ctx.save_for_backward(rhs, rhs_solves, weights, shifts, *matrix_args)

        sqrt_inv_matmul_res = (rhs_solves * weights).sum(0)
        return sqrt_inv_matmul_res

    @staticmethod
    def backward(ctx, sqrt_inv_matmul_grad):
        rhs, rhs_solves, weights, shifts, *matrix_args = ctx.saved_tensors
        rhs_grad = None
        matrix_arg_grads = [None] * len(matrix_args)

        # Compute
        grad_solves, _, _ = contour_integral_quad(
            ctx.lazy_tsr, sqrt_inv_matmul_grad, inverse=True, weights=weights, shifts=shifts
        )
        grad_solves.mul_(weights)

        # Compute lhs/rhs grads
        if ctx.needs_input_grad[1]:
            # lhs_grad term from sqrt_inv_matmul
            rhs_grad = grad_solves.sum(0)

        # Compute matrix grads
        matrix_arg_grads = ctx.lazy_tsr._quad_form_derivative(
            torch.cat([rhs_solves, grad_solves], -1), torch.cat([grad_solves, rhs_solves], -1).mul_(0.5)
        )

        res = (None, rhs_grad, *matrix_arg_grads)
        return res
