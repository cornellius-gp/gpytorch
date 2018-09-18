import torch
from .cholesky import batch_potrf, batch_potrs


def pivoted_cholesky(matrix, max_iter, error_tol=1e-3):
    from ..lazy import LazyTensor, NonLazyTensor

    # matrix is assumed to be batch_size x n x n
    if matrix.ndimension() < 3:
        batch_size = 1
        batch_mode = False
    else:
        batch_size = matrix.size(0)
        batch_mode = True
    matrix_size = matrix.size(-1)

    # Need to get diagonals. This is easy if it's a LazyTensor, since
    # LazyTensor.diag() operates in batch mode.
    if isinstance(matrix, LazyTensor):
        matrix = matrix.evaluate_kernel()
        matrix_diag = matrix._approx_diag()
    elif torch.is_tensor(matrix):
        matrix_diag = NonLazyTensor(matrix).diag()

    if not batch_mode:
        matrix_diag.unsqueeze_(0)
    # matrix_diag is now batch_size x n

    # Make sure max_iter isn't bigger than the matrix
    max_iter = min(max_iter, matrix_size)

    errors = torch.norm(matrix_diag, 1, dim=1)
    permutation = torch.arange(0, matrix_size, dtype=torch.long, device=matrix_diag.device)
    permutation = permutation.repeat(batch_size, 1)

    m = 0
    L = torch.zeros(batch_size, max_iter, matrix_size, dtype=matrix.dtype, device=matrix.device)
    full_batch_slice = torch.arange(0, batch_size, dtype=torch.long, device=permutation.device)
    while m < max_iter and torch.max(errors) > error_tol:
        permuted_diags = torch.gather(matrix_diag, 1, permutation)[:, m:]
        max_diag_values, max_diag_indices = torch.max(permuted_diags, 1)

        max_diag_indices = max_diag_indices + m

        # Swap pi_m and pi_i in each row, where pi_i is the element of the permutation
        # corresponding to the max diagonal element
        old_pi_m = permutation[:, m].clone()
        new_pi_m = permutation[full_batch_slice, max_diag_indices].clone()
        permutation[:, m] = new_pi_m
        permutation[full_batch_slice, max_diag_indices] = old_pi_m
        pi_m = permutation[:, m]

        L_m = L[:, m]  # Will be all zeros -- should we use torch.zeros?
        L_m[full_batch_slice, pi_m] = torch.sqrt(max_diag_values)

        if not batch_mode:
            row = matrix[pi_m, :]
            if row.ndimension() < 2:
                row.unsqueeze_(0)
        else:
            row = matrix[full_batch_slice, pi_m, :]

        if isinstance(row, LazyTensor):
            row = row.evaluate()

        if m + 1 < matrix_size:
            pi_i = permutation[:, m + 1 :]

            L_m_new = row.gather(1, pi_i)
            if m > 0:
                L_prev = L[:, :m].gather(2, pi_i.unsqueeze(1).repeat(1, m, 1))
                update = L[:, :m].gather(2, pi_m.unsqueeze(1).unsqueeze(1).repeat(1, m, 1))
                L_m_new -= torch.sum(update * L_prev, dim=1)

            L_m_new /= L_m.gather(1, pi_m.unsqueeze(1))
            L_m.scatter_(1, pi_i, L_m_new)

            matrix_diag_current = matrix_diag.gather(1, pi_i)
            matrix_diag.scatter_(1, pi_i, matrix_diag_current - L_m_new ** 2)
            L[:, m] = L_m

            errors = torch.norm(matrix_diag.gather(1, pi_i), 1, dim=1)
        m = m + 1

    if not batch_mode:
        return L[0, :m, :]
    else:
        return L[:, :m, :]


def woodbury_factor(low_rank_mat, shift):
    """
    Given a low rank (k x n) matrix V and a shift, returns the
    matrix R so that
        R = (I_k + 1/shift VV')^{-1}V
    to be used in solves with (V'V + shift I) via the Woodbury formula
    """
    k = low_rank_mat.size(-2)
    shifted_mat = low_rank_mat.matmul(low_rank_mat.transpose(-1, -2) / shift.unsqueeze(-1))

    shifted_mat = shifted_mat + torch.eye(k, dtype=shifted_mat.dtype, device=shifted_mat.device)

    if low_rank_mat.ndimension() == 3:
        R = batch_potrs(low_rank_mat, batch_potrf(shifted_mat))
    else:
        R = torch.potrs(low_rank_mat, shifted_mat.potrf())

    return R


def woodbury_solve(vector, low_rank_mat, woodbury_factor, shift):
    """
    Solves the system of equations:
        (sigma*I + VV')x = b
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
