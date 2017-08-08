import torch
import gpytorch.utils.fft as fft
import gpytorch.utils as utils


def index_coef_to_sparse(J, C, num_grid_points):
    num_target_points, num_coefficients = C.size()
    J_list = [[], []]
    value_list = []
    for i in range(num_target_points):
        for j in range(num_coefficients):
            if C[i, j] == 0:
                continue
            J_list[0].append(i)
            J_list[1].append(J[i, j])
            value_list.append(C[i, j])

    index_tensor = torch.LongTensor(J_list)
    value_tensor = torch.FloatTensor(value_list)
    W = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([num_target_points, num_grid_points]))
    return W


def toeplitz(toeplitz_column, toeplitz_row):
    """
    Constructs tensor version of toeplitz matrix from column vector
    Args:
        - toeplitz_column (vector n) - column of toeplitz matrix
        - toeplitz_row (vector n-1) - row of toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    assert toeplitz_column.ndimension() == 1
    assert toeplitz_row.ndimension() == 1
    assert toeplitz_column[0] == toeplitz_row[0]
    assert len(toeplitz_column) == len(toeplitz_row)
    assert type(toeplitz_column) == type(toeplitz_row)

    res = torch.Tensor(len(toeplitz_column), len(toeplitz_column)).type_as(toeplitz_column)
    for i, val in enumerate(toeplitz_column):
        for j in range(len(toeplitz_column) - i):
            res[j + i, j] = val
    for i, val in list(enumerate(toeplitz_row))[1:]:
        for j in range(len(toeplitz_row) - i):
            res[j, j + i] = val
    return res


def sym_toeplitz(toeplitz_column):
    """
    Constructs tensor version of symmetric toeplitz matrix from column vector
    Args:
        - toeplitz_column (vector n) - column of Toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    return toeplitz(toeplitz_column, toeplitz_column)


def toeplitz_getitem(toeplitz_column, toeplitz_row, i, j):
    """
    Gets the (i,j)th entry of a Toeplitz matrix T.
    Args:
        - toeplitz_column (vector n) - column of Toeplitz matrix
        - toeplitz_row (vector n) - row of Toeplitz matrix
        - i (scalar) - row of entry to get
        - j (scalar) - column of entry to get
    Returns:
        - T[i,j], where T is the Toeplitz matrix specified by c and r.
    """
    index = i - j
    if index < 0:
        return toeplitz_row[abs(index)]
    else:
        return toeplitz_column[index]


def toeplitz_mm(toeplitz_column, toeplitz_row, matrix):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n) - First row of the Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix to multiply the Toeplitz matrix with.
    Returns:
        - Matrix (n x p) - The result of the matrix-vector multiply TM.
    """
    if toeplitz_column.ndimension() != 1 or toeplitz_row.ndimension() != 1 or matrix.ndimension() != 2:
        raise RuntimeError('The first two inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                            matrix), and the last input should be a matrix.')

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if len(toeplitz_column) != len(matrix):
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a matrix with leading \
                            dimension {}.'.format(len(toeplitz_column), len(toeplitz_column), len(matrix)))

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first element, \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(toeplitz_column[0], toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(matrix):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    _, num_rhs = matrix.size()
    orig_size = len(toeplitz_column)
    r_reverse = utils.reverse(toeplitz_row[1:])
    toeplitz_column.resize_(orig_size + len(r_reverse))
    toeplitz_column[orig_size:].copy_(r_reverse)

    matrix.resize_(2 * orig_size - 1, num_rhs)
    matrix[orig_size:, :].fill_(0)

    fft_M = fft.fft1(matrix.t().contiguous())
    fft_c = fft.fft1(toeplitz_column).expand_as(fft_M)
    fft_product = torch.zeros(fft_M.size())

    fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
    fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

    res = fft.ifft1(fft_product, (num_rhs, 2 * orig_size - 1)).t()
    toeplitz_column.resize_(orig_size)
    toeplitz_row.resize_(orig_size)
    matrix.resize_(orig_size, num_rhs)
    res = res[:orig_size, :]
    return res


def toeplitz_mv(toeplitz_column, toeplitz_row, vector):
    """
    Performs a matrix-vector multiplication Tv where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n) - First row of the Toeplitz matrix T.
        - vector (vector n) - Vector to multiply the Toeplitz matrix with.
    Returns:
        - vector n - The result of the matrix-vector multiply Tv.
    """
    if toeplitz_column.ndimension() != 1 or toeplitz_row.ndimension() != 1 or vector.ndimension() != 1:
        raise RuntimeError('All inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector vector).')

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if len(toeplitz_column) != len(vector):
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a length \
                            {} vector.'.format(len(toeplitz_column), len(toeplitz_column), len(vector)))

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(toeplitz_column[0], toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(vector):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    orig_size = len(toeplitz_column)
    r_reverse = utils.reverse(toeplitz_row[1:])
    toeplitz_column.resize_(orig_size + len(r_reverse))
    toeplitz_column[orig_size:].copy_(r_reverse)

    vector.resize_(2 * orig_size - 1)
    vector[orig_size:].fill_(0)

    fft_c = fft.fft1(toeplitz_column)
    fft_v = fft.fft1(vector)
    fft_product = torch.zeros(fft_c.size())

    fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
    fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
    fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
    fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

    res = fft.ifft1(fft_product, toeplitz_column.size())
    toeplitz_column.resize_(orig_size)
    toeplitz_row.resize_(orig_size)
    vector.resize_(orig_size)
    res.resize_(orig_size)
    return res


def interpolated_toeplitz_mul(c, vector, W_left=None, W_right=None, noise_diag=None):
    """
    Given a interpolated symmetric Toeplitz matrix W_left*T*W_right, plus possibly an additional
    diagonal component s*I, compute a matrix-vector product with some vector or matrix vector.

    Args:
        - c (vector matrix) - First column of the symmetric Toeplitz matrix T
        - W_left (sparse matrix nxm) - Left interpolation matrix
        - W_right (sparse matrix pxm) - Right interpolation matrix
        - vector (matrix pxk) - Vector (k=1) or matrix (k>1) to multiply WTW with
        - noise_diag (vector p) - If not none, add (s*I)vector to WTW at the end.

    Returns:
        - matrix nxk - The result of multiplying (WTW + sI)vector if noise_diag exists, or (WTW)vector otherwise.
    """
    noise_term = None
    if vector.ndimension() == 1:
        if noise_diag is not None:
            noise_term = noise_diag.expand_as(vector) * vector
        vector = vector.unsqueeze(1)
        mul_func = utils.toeplitz.toeplitz_mv
    else:
        if noise_diag is not None:
            noise_term = noise_diag.unsqueeze(1).expand_as(vector) * vector
        mul_func = utils.toeplitz.toeplitz_mm

    if W_left is not None:
        # Get W_{r}^{T}vector
        Wt_times_v = torch.dsmm(W_right.t(), vector)
        # Get (TW_{r}^{T})vector
        TWt_v = mul_func(c, c, Wt_times_v.squeeze())

        if TWt_v.ndimension() == 1:
            TWt_v.unsqueeze_(1)

        # Get (W_{l}TW_{r}^{T})vector
        WTWt_v = torch.dsmm(W_left, TWt_v).squeeze()
    else:
        WTWt_v = mul_func(c, c, vector)

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)vector
        WTWt_v = WTWt_v + noise_term

    return WTWt_v
