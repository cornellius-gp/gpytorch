#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings
from ..utils.contour_integral_quad import contour_integral_quad


class SqrtInvMatmul(Function):
    @staticmethod
    def forward(ctx, representation_tree, *args):
        ctx.representation_tree = representation_tree
        rhs, lhs, *matrix_args = args
        ctx.lazy_tsr = ctx.representation_tree(*matrix_args)

        terms = torch.cat([rhs, lhs.transpose(-1, -2)], dim=-1)
        solves, weights, no_shift_solves = contour_integral_quad(
            ctx.lazy_tsr, terms, inverse=True, num_contour_quadrature=settings.num_contour_quadrature.value()
        )
        rhs_solves, lhs_solves = solves.split([rhs.size(-1), lhs.size(-2)], dim=-1)
        lhs_no_shift_solves = no_shift_solves[..., -lhs.size(-2) :]
        ctx.save_for_backward(rhs, lhs, rhs_solves, lhs_solves, weights, lhs_no_shift_solves, *matrix_args)

        sqrt_inv_matmul_res = lhs @ (rhs_solves * weights).sum(0)
        inv_quad_res = (lhs_no_shift_solves.transpose(-1, -2) * lhs).sum(dim=-1).mul_(-1)

        # Record some stats on how good the solves are
        if settings.record_ciq_stats.on():
            with torch.no_grad():
                settings.record_ciq_stats.ciq_diff = (
                    ((lhs_solves * weights).sum(dim=0).pow(2).sum(dim=-2).sub_(inv_quad_res))
                    .div_(inv_quad_res.clamp_min_(1e-5))
                    .abs_()
                    .mean()
                    .item()
                )

        return sqrt_inv_matmul_res, inv_quad_res

    @staticmethod
    def backward(ctx, sqrt_inv_matmul_grad, inv_quad_grad):
        rhs, lhs, rhs_solves, lhs_solves, weights, lhs_no_shift_solves, *matrix_args = ctx.saved_tensors
        rhs_grad = None
        lhs_grad = None
        matrix_arg_grads = [None] * len(matrix_args)

        # Intermediate terms for sqrt_inv_matmul
        weighted_rhs_solves_mul_grad = rhs_solves.mul(weights) @ sqrt_inv_matmul_grad.transpose(-1, -2)

        # Intermediate terms for quad
        neg_inv_quad_solves_mul_grad = lhs_no_shift_solves.mul(inv_quad_grad.unsqueeze(-2)).mul(-1)

        # Compute lhs/rhs grads
        if ctx.needs_input_grad[1]:
            # lhs_grad term from sqrt_inv_matmul
            lhs_grad = weighted_rhs_solves_mul_grad.transpose(-1, -2).sum(0)
            # lhs_grad term from inv_quad
            lhs_grad.add_(2, neg_inv_quad_solves_mul_grad.transpose(-1, -2))
        if ctx.needs_input_grad[2]:
            rhs_grad = (lhs_solves @ sqrt_inv_matmul_grad).mul(weights).sum(0)

        # Compute matrix grads
        terms1 = torch.cat([lhs_no_shift_solves.unsqueeze(0), lhs_solves], 0)
        terms2 = torch.cat([neg_inv_quad_solves_mul_grad.unsqueeze(0), weighted_rhs_solves_mul_grad], 0)
        matrix_arg_grads = ctx.lazy_tsr._quad_form_derivative(
            torch.cat([terms1, terms2], -1), torch.cat([terms2, terms1], -1).mul_(0.5)
        )

        res = (None, rhs_grad, lhs_grad, *matrix_arg_grads)
        return res
