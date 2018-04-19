import torch


def pivoted_cholesky(get_item_closure, matrix_diag, max_iter, error_tol=1e-3,
                     tensor_cls=None, batch_size=None, matrix_size=None):
    """
        get_indices_closure - either the matrix itself, or a function to get a list of elements
            from a matrix
        max_iter - maximum number of iterations to run
        error_tol - tolerance to error out on

        tensor_cls - class of tensor (doesn't need to be defined if get_indices_closure is a matrix)
        batch_size - how many matrices in batch (None means non-batch) (doesn't need
            to be defined if get_indices_closure is a matrix)
        matrix_size - number of rows/columns of matrix (doesn't need to be defined if get_indices_closure is a matrix)
    """
    batch_mode = False
    if matrix_diag.ndimension() == 1:
        matrix_diag = matrix_diag.unsqueeze(0)

    # Make sure get_indices_closure is a function, and all arguments are defined
    if torch.is_tensor(get_item_closure):
        matrix = get_item_closure

        def default_getitem_closure(*args):
            return matrix[args]

        get_item_closure = default_getitem_closure
        tensor_cls = matrix.new
        matrix_size = matrix.size(-1)
        if matrix.ndimension() == 3:
            batch_size = matrix.size(0)
            batch_mode = True
        else:
            batch_size = 1

    else:
        if tensor_cls is None:
            raise RuntimeError('tensor_cls must be defined')
        if matrix_size is None:
            raise RuntimeError('matrix_size must be defined')
        if batch_size is not None:
            batch_mode = True
        else:
            batch_size = 1

    # Make sure max_iter isn't bigger than the matrix
    max_iter = min(max_iter, matrix_size)

    # Define matrix which stores the permutation
    permutation = tensor_cls(matrix_size).long()
    torch.arange(0, matrix_size, out=permutation)
    permutation = permutation.repeat(batch_size, 1)

    # Define a matrix for slicing into a batch
    full_batch_slice = permutation.new(batch_size)
    torch.arange(batch_size, out=full_batch_slice)

    # The result will be stored here
    # TODO: pivoted_cholesky should take tensor_cls and use that here instead
    L = tensor_cls(batch_size, max_iter, matrix_size).zero_()

    # Compute the errors
    errors = torch.norm(matrix_diag, 1, dim=1)

    # Start the iteration
    m = 0
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
            row = get_item_closure([pi_m])
        else:
            row = get_item_closure([full_batch_slice, pi_m, slice(None)])

        if m + 1 < matrix_size:
            pi_i = permutation[:, m + 1:]
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
    shifted_mat = (1 / shift) * low_rank_mat.matmul(low_rank_mat.transpose(-1, -2))

    shifted_mat = shifted_mat + shifted_mat.new(k).fill_(1).diag()

    if low_rank_mat.ndimension() == 3:
        R = torch.cat([torch.potrs(low_rank_mat[i], shifted_mat[i].potrf()).unsqueeze(0)
                       for i in range(shifted_mat.size(0))])
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
        - shift (scalar) - shift value sigma
    """
    right = (1 / shift) * low_rank_mat.transpose(-1, -2).matmul(woodbury_factor.matmul(vector))
    return (1 / shift) * (vector - right)
