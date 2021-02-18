#!/usr/bin/env python3

import torch

from .. import settings


def pivoted_cholesky(matrix, max_iter, error_tol=None):
    from ..lazy import lazify, LazyTensor

    batch_shape = matrix.shape[:-2]
    matrix_shape = matrix.shape[-2:]

    if error_tol is None:
        error_tol = settings.preconditioner_tolerance.value()

    # Need to get diagonals. This is easy if it's a LazyTensor, since
    # LazyTensor.diag() operates in batch mode.
    matrix = lazify(matrix)
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

    # Get batch indices
    batch_iters = [
        torch.arange(0, size, dtype=torch.long, device=matrix_diag.device)
        .unsqueeze_(-1)
        .repeat(torch.Size(batch_shape[:i]).numel(), torch.Size(batch_shape[i + 1 :]).numel())
        .view(-1)
        for i, size in enumerate(batch_shape)
    ]

    # Maybe log
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(
            f"Running Pivoted Cholesky on a {matrix.shape} RHS for {max_iter} iterations."
        )

    m = 0
    while (m == 0) or (m < max_iter and torch.max(errors) > error_tol):
        permuted_diags = torch.gather(matrix_diag, -1, permutation[..., m:])
        max_diag_values, max_diag_indices = torch.max(permuted_diags, -1)

        max_diag_indices = max_diag_indices + m

        # Swap pi_m and pi_i in each row, where pi_i is the element of the permutation
        # corresponding to the max diagonal element
        old_pi_m = permutation[..., m].clone()
        permutation[..., m].copy_(permutation.gather(-1, max_diag_indices.unsqueeze(-1)).squeeze_(-1))
        permutation.scatter_(-1, max_diag_indices.unsqueeze(-1), old_pi_m.unsqueeze(-1))
        pi_m = permutation[..., m].contiguous()

        L_m = L[..., m, :]  # Will be all zeros -- should we use torch.zeros?
        L_m.scatter_(-1, pi_m.unsqueeze(-1), max_diag_values.sqrt().unsqueeze_(-1))

        row = matrix[(*batch_iters, pi_m.view(-1), slice(None, None, None))]
        if isinstance(row, LazyTensor):
            row = row.evaluate()
        row = row.view(*batch_shape, matrix_shape[-1])

        if m + 1 < matrix_shape[-1]:
            pi_i = permutation[..., m + 1 :].contiguous()

            L_m_new = row.gather(-1, pi_i)
            if m > 0:
                L_prev = L[..., :m, :].gather(-1, pi_i.unsqueeze(-2).repeat(*(1 for _ in batch_shape), m, 1))
                update = L[..., :m, :].gather(-1, pi_m.view(*pi_m.shape, 1, 1).repeat(*(1 for _ in batch_shape), m, 1))
                L_m_new -= torch.sum(update * L_prev, dim=-2)

            L_m_new /= L_m.gather(-1, pi_m.unsqueeze(-1))
            L_m.scatter_(-1, pi_i, L_m_new)

            matrix_diag_current = matrix_diag.gather(-1, pi_i)
            matrix_diag.scatter_(-1, pi_i, matrix_diag_current - L_m_new ** 2)
            L[..., m, :] = L_m

            errors = torch.norm(matrix_diag.gather(-1, pi_i), 1, dim=-1) / orig_error
        m = m + 1

    return L[..., :m, :].transpose(-1, -2).contiguous()
