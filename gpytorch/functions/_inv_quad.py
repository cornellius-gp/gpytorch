#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings


def _solve(lazy_tsr, rhs):
    if (
        settings.fast_computations.solves.off()
        or settings.fast_computations.log_prob.off()
        or lazy_tsr.size(-1) <= settings.max_cholesky_size.value()
    ):
        return lazy_tsr.cholesky()._cholesky_solve(rhs)
    else:
        with torch.no_grad():
            preconditioner = lazy_tsr.detach()._inv_matmul_preconditioner()
        return lazy_tsr._solve(rhs, preconditioner)


class InvQuad(Function):
    """
    Given a PSD matrix A (or a batch of PSD matrices A), this function computes b A^{-1} b
    where b is a vector or batch of vectors
    """

    @staticmethod
    def forward(ctx, representation_tree, *args):
        """
        *args - The arguments representing the PSD matrix A (or batch of PSD matrices A)
        If inv_quad is true, the first entry in *args is inv_quad_rhs (Tensor)
        - the RHS of the matrix solves.

        Returns:
        - (Scalar) The inverse quadratic form (or None, if inv_quad is False)
        - (Scalar) The log determinant (or None, if logdet is False)
        """
        inv_quad_rhs, *matrix_args = args
        ctx.representation_tree = representation_tree
        # Get closure for matmul
        lazy_tsr = ctx.representation_tree(*matrix_args)

        # RHS for inv_quad
        ctx.is_vector = False
        if inv_quad_rhs.ndimension() == 1:
            inv_quad_rhs = inv_quad_rhs.unsqueeze(-1)
            ctx.is_vector = True

        # Perform solves (for inv_quad) and tridiagonalization (for estimating logdet)
        inv_quad_solves = _solve(lazy_tsr, inv_quad_rhs)
        inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)

        to_save = matrix_args + [inv_quad_solves]
        ctx.save_for_backward(*to_save)

        if settings.memory_efficient.off():
            ctx._lazy_tsr = lazy_tsr

        return inv_quad_term

    @staticmethod
    def backward(ctx, inv_quad_grad_output):
        *matrix_args, inv_quad_solves = ctx.saved_tensors

        if hasattr(ctx, "_lazy_tsr"):
            lazy_tsr = ctx._lazy_tsr
        else:
            lazy_tsr = ctx.representation_tree(*matrix_args)

        # Fix grad_output sizes
        inv_quad_grad_output = inv_quad_grad_output.unsqueeze(-2)
        neg_inv_quad_solves_times_grad_out = inv_quad_solves.mul(inv_quad_grad_output).mul(-1)

        matrix_arg_grads = [None] * len(matrix_args)

        # input_1 gradient
        if any(ctx.needs_input_grad[2:]):
            left_factors = neg_inv_quad_solves_times_grad_out
            right_factors = inv_quad_solves
            matrix_arg_grads = lazy_tsr._quad_form_derivative(left_factors, right_factors)

        # input_2 gradients
        if ctx.needs_input_grad[1]:
            inv_quad_rhs_grad = neg_inv_quad_solves_times_grad_out.mul(-2)
        else:
            inv_quad_rhs_grad = torch.zeros_like(inv_quad_solves)
        if ctx.is_vector:
            inv_quad_rhs_grad.squeeze_(-1)

        res = tuple([None] + [inv_quad_rhs_grad] + list(matrix_arg_grads))
        return tuple(res)
