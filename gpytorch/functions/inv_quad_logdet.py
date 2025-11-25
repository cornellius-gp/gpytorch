#!/usr/bin/env python3

import torch

from torch import Tensor


class TensorInvQuadLogdet(torch.autograd.Function):
    r"""This function computes the inverse quadratic form and the log determinant of a positive semi-definite matrix.
    This is a light weight implementation of `LinearOperator.inv_quad_logdet`. The main motivation is to avoid the
    overhead of linear operators for dense kernel matrices by doing linear algebra operations directly on torch tensors.
    """

    @staticmethod
    def forward(
        ctx,
        matrix: Tensor,
        inv_quad_rhs: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the inverse quadratic form and the log determinant.

        :param matrix: A positive semi-definite matrix of size `(..., N, N)`.
        :param inv_quad_rhs: The right-hand side vector of size `(..., N, 1)`.
        :return: The inverse quadratic form and the log determinant, both of size `(...)`.
        """
        chol = torch.linalg.cholesky(matrix)

        # The inverse quadratic term
        inv_quad_solves = torch.cholesky_solve(inv_quad_rhs, chol)
        inv_quad_term = (inv_quad_solves * inv_quad_rhs).sum(-2)
        inv_quad_term = inv_quad_term.squeeze(-1)

        # The log determinant term
        logdet_term = 2.0 * chol.diagonal(dim1=-1, dim2=-2).log().sum(-1)

        ctx.save_for_backward(chol, inv_quad_solves)

        return inv_quad_term, logdet_term

    @staticmethod
    def backward(ctx, d_inv_quad: Tensor, d_logdet: Tensor) -> tuple[Tensor, Tensor]:
        r"""Compute the backward pass for the inverse quadratic form and the log determinant.

        :param d_inv_quad: The gradient of the inverse quadratic form of size `(...)`.
        :param d_logdet: The gradient of the log determinant of size `(...)`.
        :return: The gradients with respect to the input matrix and the right-hand side vector.
        """
        chol, inv_quad_solves = ctx.saved_tensors

        d_matrix_one = (
            -1.0 * inv_quad_solves @ inv_quad_solves.transpose(-2, -1) * d_inv_quad.unsqueeze(-1).unsqueeze(-1)
        )
        d_matrix_two = torch.cholesky_inverse(chol) * d_logdet.unsqueeze(-1).unsqueeze(-1)
        d_matrix = d_matrix_one + d_matrix_two

        d_inv_quad_rhs = 2.0 * inv_quad_solves * d_inv_quad.unsqueeze(-1).unsqueeze(-1)

        return d_matrix, d_inv_quad_rhs
