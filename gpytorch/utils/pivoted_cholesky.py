#!/usr/bin/env python3

import torch


def pivoted_cholesky(matrix, max_iter, error_tol=1e-3):
    from ..lazy import LazyTensor, NonLazyTensor

    batch_shape = matrix.shape[:-2]
    matrix_shape = matrix.shape[-2:]

    # Need to get diagonals. This is easy if it's a LazyTensor, since
    # LazyTensor.diag() operates in batch mode.
    if isinstance(matrix, LazyTensor):
        matrix = matrix.evaluate_kernel()
        matrix_diag = matrix._approx_diag()
    elif torch.is_tensor(matrix):
        matrix_diag = NonLazyTensor(matrix).diag()

    # Make sure max_iter isn't bigger than the matrix
    max_iter = min(max_iter, matrix_shape[-1])

    # What we're returning
    L = torch.zeros(*batch_shape, max_iter, matrix_shape[-1], dtype=matrix.dtype, device=matrix.device)
    errors = torch.norm(matrix_diag, 1, dim=-1)

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

    m = 0
    while m < max_iter and torch.max(errors) > error_tol:
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

            errors = torch.norm(matrix_diag.gather(-1, pi_i), 1, dim=-1)
        m = m + 1

    return L[..., :m, :].contiguous()


def woodbury_factor(low_rank_mat, shift):
    r"""
    Given a low rank (k x n) matrix V and a shift, returns the
    matrix R so that

    .. math::

        \begin{equation*}
            R = (I_k + 1/shift VV')^{-1}V
        \end{equation*}

    to be used in solves with (V'V + shift I) via the Woodbury formula
    """
    k = low_rank_mat.size(-2)
    shifted_mat = low_rank_mat.matmul(low_rank_mat.transpose(-1, -2) / shift.unsqueeze(-1))

    shifted_mat = shifted_mat + torch.eye(k, dtype=shifted_mat.dtype, device=shifted_mat.device)

    R = torch.potrs(low_rank_mat, torch.cholesky(shifted_mat, upper=True))
    return R


def woodbury_solve(vector, low_rank_mat, woodbury_factor, shift):
    """
    Solves the system of equations: :math:`(sigma*I + VV')x = b`
    Using the Woodbury formula.

    Input:
        - vector (size n) - right hand side vector b to solve with.
        - woodbury_factor (k x n) - The result of calling woodbury_factor on V
          and the shift, \sigma
        - shift (vector) - shift value sigma
    """
    if vector.ndimension() > 1:
        shift = shift.unsqueeze(-1)

    right = low_rank_mat.transpose(-1, -2).matmul(woodbury_factor.matmul(vector / shift))
    return (vector - right) / shift
