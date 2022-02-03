#!/usr/bin/env python3

import torch
from torch.autograd import Function

from .. import settings
from ..utils.cholesky import psd_safe_cholesky
from ..utils.permutation import apply_permutation, inverse_permutation


class PivotedCholesky(Function):
    @staticmethod
    def forward(ctx, representation_tree, max_iter, error_tol, *matrix_args):
        ctx.representation_tree = representation_tree
        matrix = ctx.representation_tree(*matrix_args)
        batch_shape = matrix.shape[:-2]
        matrix_shape = matrix.shape[-2:]

        if error_tol is None:
            error_tol = settings.preconditioner_tolerance.value()

        # Need to get diagonals. This is easy if it's a LazyTensor, since
        # LazyTensor.diag() operates in batch mode.
        matrix_diag = matrix._approx_diag()

        # Make sure max_iter isn't bigger than the matrix
        max_iter = min(max_iter, matrix_shape[-1])

        # What we're returning
        L = torch.zeros(*batch_shape, max_iter, matrix_shape[-1], dtype=matrix.dtype, device=matrix.device)
        orig_error = torch.max(matrix_diag, dim=-1)[0]
        errors = torch.norm(matrix_diag, 1, dim=-1) / orig_error

        # The permutation
        permutation = torch.arange(0, matrix_shape[-1], dtype=torch.long, device=matrix_diag.device)
        permutation = permutation.repeat(*batch_shape, 1)

        # Maybe log
        if settings.verbose_linalg.on():
            settings.verbose_linalg.logger.debug(
                f"Running Pivoted Cholesky on a {matrix.shape} RHS for {max_iter} iterations."
            )

        m = 0
        while (m == 0) or (m < max_iter and torch.max(errors) > error_tol):
            # Get the maximum diagonal value and index
            # This will serve as the next diagonal entry of the Cholesky,
            # as well as the next entry in the permutation matrix
            permuted_diags = torch.gather(matrix_diag, -1, permutation[..., m:])
            max_diag_values, max_diag_indices = torch.max(permuted_diags, -1)
            max_diag_indices = max_diag_indices + m

            # Swap pi_m and pi_i in each row, where pi_i is the element of the permutation
            # corresponding to the max diagonal element
            old_pi_m = permutation[..., m].clone()
            permutation[..., m].copy_(permutation.gather(-1, max_diag_indices.unsqueeze(-1)).squeeze_(-1))
            permutation.scatter_(-1, max_diag_indices.unsqueeze(-1), old_pi_m.unsqueeze(-1))
            pi_m = permutation[..., m].contiguous()

            # Populate L[..., m, m] with the sqrt of the max diagonal element
            L_m = L[..., m, :]  # Will be all zeros -- should we use torch.zeros?
            L_m.scatter_(-1, pi_m.unsqueeze(-1), max_diag_values.sqrt().unsqueeze_(-1))

            # Populater L[... m:, m] with L[..., m:, m] * L[..., m, m].sqrt()
            if m + 1 < matrix_shape[-1]:
                # Get next row of the permuted matrix
                row = apply_permutation(matrix, pi_m.unsqueeze(-1), right_permutation=None).squeeze(-2)
                pi_i = permutation[..., m + 1 :].contiguous()

                L_m_new = row.gather(-1, pi_i)
                if m > 0:
                    L_prev = L[..., :m, :].gather(-1, pi_i.unsqueeze(-2).repeat(*(1 for _ in batch_shape), m, 1))
                    update = L[..., :m, :].gather(
                        -1, pi_m.view(*pi_m.shape, 1, 1).repeat(*(1 for _ in batch_shape), m, 1)
                    )
                    L_m_new -= torch.sum(update * L_prev, dim=-2)

                L_m_new /= L_m.gather(-1, pi_m.unsqueeze(-1))
                L_m.scatter_(-1, pi_i, L_m_new)

                matrix_diag_current = matrix_diag.gather(-1, pi_i)
                matrix_diag.scatter_(-1, pi_i, matrix_diag_current - L_m_new ** 2)
                L[..., m, :] = L_m

                # Keep track of errors - for potential early stopping
                errors = torch.norm(matrix_diag.gather(-1, pi_i), 1, dim=-1) / orig_error

            m = m + 1

        # Save items for backward pass, and return output
        ctx.save_for_backward(permutation, permutation[..., :m], *matrix_args)
        return L[..., :m, :].transpose(-1, -2).contiguous(), permutation

    def backward(ctx, grad_output, _):
        full_permutation, short_permutation, *_matrix_args = ctx.saved_tensors
        _inverse_permutation = inverse_permutation(full_permutation)
        m = short_permutation.size(-1)  # The rank of the pivoted Cholesky factor

        with torch.enable_grad():
            # Create a new set of matrix args that we can backpropagate through
            matrix_args = [matrix_arg.detach().requires_grad_(True) for matrix_arg in _matrix_args]

            # Create new linear operator using new matrix args
            matrix = ctx.representation_tree(*matrix_args)

            # Get Krows - rows of the matrix that were selected for pivoted Cholesky
            Krows = apply_permutation(matrix, full_permutation, short_permutation)

            # Get L - Cholesky factor of K[..., :m, :m]
            L = psd_safe_cholesky(Krows[..., :m, :])

            # Compute (Krows * L^{-T}) - the (pivoted) result of Pivoted Cholesky
            res_pivoted = torch.cat(
                [L, torch.triangular_solve(Krows[..., m:, :].transpose(-1, -2), L, upper=False)[0].transpose(-1, -2)],
                dim=-2,
            )
            res = apply_permutation(res_pivoted, left_permutation=_inverse_permutation, right_permutation=None)

            # Now compute the backward pass of res
            res.backward(gradient=grad_output)

        # Define gradient placeholders
        return tuple([None, None, None] + [matrix_arg.grad for matrix_arg in matrix_args])
