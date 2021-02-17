#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings, utils


class SqrtInvMatmul(Function):
    """
    Given a PD matrix A, this function computes one of the following:
    - A^{-1/2} rhs
    - lhs A^{-1/2} rhs
    using contour integral quadrature.
    """

    @staticmethod
    def forward(ctx, representation_tree, rhs, lhs, *matrix_args):
        ctx.representation_tree = representation_tree
        ctx.lazy_tsr = ctx.representation_tree(*matrix_args)

        if lhs is not None:
            terms = torch.cat([rhs, lhs.transpose(-1, -2)], dim=-1)
            solves, weights, no_shift_solves, shifts = utils.contour_integral_quad(
                ctx.lazy_tsr, terms, inverse=True, num_contour_quadrature=settings.num_contour_quadrature.value()
            )
            rhs_solves, lhs_solves = solves.split([rhs.size(-1), lhs.size(-2)], dim=-1)
            lhs_no_shift_solves = no_shift_solves[..., -lhs.size(-2) :]
            sqrt_inv_matmul_res = lhs @ (rhs_solves * weights).sum(0)
            inv_quad_res = (lhs_no_shift_solves.transpose(-1, -2) * lhs).sum(dim=-1).mul_(-1)
        else:
            rhs_solves, weights, _, shifts = utils.contour_integral_quad(
                ctx.lazy_tsr, rhs, inverse=True, num_contour_quadrature=settings.num_contour_quadrature.value()
            )
            sqrt_inv_matmul_res = (rhs_solves * weights).sum(0)
            lhs_solves = None
            lhs_no_shift_solves = None
            inv_quad_res = torch.zeros(ctx.lazy_tsr.batch_shape, dtype=rhs.dtype, device=rhs.device)

        # Save for backwards
        ctx.save_for_backward(rhs, lhs, rhs_solves, lhs_solves, weights, shifts, lhs_no_shift_solves, *matrix_args)

        return sqrt_inv_matmul_res, inv_quad_res

    @staticmethod
    def backward(ctx, sqrt_inv_matmul_grad, inv_quad_grad):
        rhs, lhs, rhs_solves, lhs_solves, weights, shifts, lhs_no_shift_solves, *matrix_args = ctx.saved_tensors
        rhs_grad = None
        lhs_grad = None
        matrix_arg_grads = [None] * len(matrix_args)

        if lhs is not None:
            # Intermediate terms for sqrt_inv_matmul/quad
            weighted_rhs_solves_mul_grad = rhs_solves.mul(weights) @ sqrt_inv_matmul_grad.transpose(-1, -2)
            neg_inv_quad_solves_mul_grad = lhs_no_shift_solves.mul(inv_quad_grad.unsqueeze(-2)).mul(-1)

            # Compute lhs grads
            if ctx.needs_input_grad[2] and lhs is not None:
                # lhs_grad term from sqrt_inv_matmul
                lhs_grad = weighted_rhs_solves_mul_grad.transpose(-1, -2).sum(0)
                # lhs_grad term from inv_quad
                lhs_grad.add_(neg_inv_quad_solves_mul_grad.transpose(-1, -2), alpha=2)

            # Compute rhs grad
            if ctx.needs_input_grad[1]:
                rhs_grad = (lhs_solves @ sqrt_inv_matmul_grad).mul(weights).sum(0)

            # Compute matrix grads
            terms1 = torch.cat([lhs_no_shift_solves.unsqueeze(0), lhs_solves], 0)
            terms2 = torch.cat([neg_inv_quad_solves_mul_grad.unsqueeze(0), weighted_rhs_solves_mul_grad], 0)
            matrix_arg_grads = ctx.lazy_tsr._quad_form_derivative(
                torch.cat([terms1, terms2], -1), torch.cat([terms2, terms1], -1).mul_(0.5)
            )

        else:
            # Intermediate terms for sqrt_inv_matmul/quad
            grad_solves, _, _, _ = utils.contour_integral_quad(
                ctx.lazy_tsr,
                sqrt_inv_matmul_grad,
                inverse=True,
                weights=weights,
                shifts=shifts,
                num_contour_quadrature=settings.num_contour_quadrature.value(),
            )
            grad_solves_mul_weights = grad_solves.mul(weights)

            # No lhs grad
            lhs_grad = None

            # Compute rhs grad
            if ctx.needs_input_grad[1]:
                rhs_grad = grad_solves_mul_weights.sum(0)

            # Compute matrix grads
            terms1 = grad_solves_mul_weights
            terms2 = rhs_solves
            matrix_arg_grads = ctx.lazy_tsr._quad_form_derivative(
                torch.cat([terms1, terms2], -1), torch.cat([terms2, terms1], -1).mul_(0.5)
            )

        res = (None, rhs_grad, lhs_grad, *matrix_arg_grads)
        return res
