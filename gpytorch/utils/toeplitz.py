import torch
import gpytorch.utils.fft as fft
import gpytorch.utils as utils


def index_coef_to_sparse(index_matrix, value_matrix, row_length):
    """
    Converts a sparse matrix stored densely in an index matrix and a value matrix
    to a torch sparse matrix.
    Args:
        - index_matrix (Matrix n-by-nz) -- A matrix that describes, for each row of the
                       sparse matrix W, which elements are nonzero.
        - value_matrix (Matrix n-by-nz) -- A matrix that describes, for each row of the
                       sparse matrix W, the values of the nonzero elements. Each value is
                       placed in the position described by the index matrix (see the
                       definition of W given below).
        - row_length (scalar) -- The size of the second dimension of W
    Returns:
        - W (Sparse matrix n-by-row_length) - A torch sparse matrix W so that
            W[i, index_matrix[i, j]] = value_matrix[i, j].
    """
    num_target_points, num_coefficients = value_matrix.size()
    index_list = [[], []]
    value_list = []
    for i in range(num_target_points):
        for j in range(num_coefficients):
            if value_matrix[i, j] == 0:
                continue
            index_list[0].append(i)
            index_list[1].append(index_matrix[i, j])
            value_list.append(value_matrix[i, j])

    index_tensor = torch.LongTensor(index_list)
    value_tensor = torch.FloatTensor(value_list)
    W = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([num_target_points, row_length]))
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
    if toeplitz_column.ndimension() != 1:
        raise RuntimeError('toeplitz_column must be a vector.')

    if toeplitz_row.ndimension() != 1:
        raise RuntimeError('toeplitz_row must be a vector.')

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(toeplitz_column[0], toeplitz_row[0]))

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if type(toeplitz_column) != type(toeplitz_row):
        raise RuntimeError('toeplitz_column and toeplitz_row should be the same type.')

    if len(toeplitz_column) == 1:
        return toeplitz_column.view(1, 1)

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


def sym_toeplitz_getitem(toeplitz_column, i, j):
    """
    Gets the (i,j)th entry of a symmetric Toeplitz matrix T.
    Args:
        - toeplitz_column (vector n) - column of symmetric Toeplitz matrix
        - i (scalar) - row of entry to get
        - j (scalar) - column of entry to get
    Returns:
        - T[i,j], where T is the Toeplitz matrix specified by c and r.
    """
    return toeplitz_getitem(toeplitz_column, toeplitz_column, i, j)


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
        raise RuntimeError('The first two inputs to ToeplitzMV should be vectors \
                            (first column c and row r of the Toeplitz matrix), and the last input should be a matrix.')

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if len(toeplitz_column) != len(matrix):
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a matrix with \
                            leading dimension {}.'.format(len(toeplitz_column), len(toeplitz_column), len(matrix)))

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first element, \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(toeplitz_column[0], toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(matrix):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    if len(toeplitz_column) == 1:
        return (toeplitz_column.view(1, 1).mm(matrix))

    _, num_rhs = matrix.size()
    orig_size = len(toeplitz_column)
    r_reverse = utils.reverse(toeplitz_row[1:])

    c_r_rev = torch.zeros(orig_size + len(r_reverse))
    c_r_rev[:orig_size] = toeplitz_column
    c_r_rev[orig_size:] = r_reverse

    temp_matrix = torch.zeros(2 * orig_size - 1, num_rhs)
    temp_matrix[:orig_size, :] = matrix

    fft_M = fft.fft1(temp_matrix.t().contiguous())
    fft_c = fft.fft1(c_r_rev).expand_as(fft_M)
    fft_product = torch.zeros(fft_M.size())

    fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
    fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

    res = fft.ifft1(fft_product, (num_rhs, 2 * orig_size - 1)).t()
    res = res[:orig_size, :]
    return res


def sym_toeplitz_mm(toeplitz_column, matrix):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix to multiply the Toeplitz matrix with.
    Returns:
        - Matrix (n x p) - The result of the matrix-vector multiply TM.
    """
    return toeplitz_mm(toeplitz_column, toeplitz_column, matrix)


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

    if len(toeplitz_column) == 1:
        return (toeplitz_column.view(1, 1).mv(vector))

    orig_size = len(toeplitz_column)
    r_reverse = utils.reverse(toeplitz_row[1:])

    c_r_rev = torch.zeros(orig_size + len(r_reverse))
    c_r_rev[:orig_size] = toeplitz_column
    c_r_rev[orig_size:] = r_reverse

    temp_vector = torch.zeros(2 * orig_size - 1)
    temp_vector[:orig_size] = vector

    fft_c = fft.fft1(c_r_rev)
    fft_v = fft.fft1(temp_vector)
    fft_product = torch.zeros(fft_c.size())

    fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
    fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
    fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
    fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

    res = fft.ifft1(fft_product, temp_vector.size())
    res.resize_(orig_size)
    return res


def sym_toeplitz_mv(toeplitz_column, vector):
    """
    Performs a matrix-vector multiplication Tv where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - vector (matrix n) - vector to multiply the Toeplitz matrix with.
    Returns:
        - vector (n) - The result of the matrix-vector multiply Tv.
    """
    return toeplitz_mv(toeplitz_column, toeplitz_column, vector)


def interpolated_sym_toeplitz_mul(toeplitz_column, vector, W_left=None, W_right=None, noise_diag=None):
    """
    Given a interpolated symmetric Toeplitz matrix W_left*T*W_right, plus possibly an additional
    diagonal component s*I, compute a product with some vector or matrix vector.

    Args:
        - toeplitz_column (vector matrix) - First column of the symmetric Toeplitz matrix T
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
        TWt_v = mul_func(toeplitz_column, toeplitz_column, Wt_times_v.squeeze())

        if TWt_v.ndimension() == 1:
            TWt_v.unsqueeze_(1)

        # Get (W_{l}TW_{r}^{T})vector
        WTWt_v = torch.dsmm(W_left, TWt_v).squeeze()
    else:
        WTWt_v = mul_func(toeplitz_column, toeplitz_column, vector)

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)vector
        WTWt_v = WTWt_v + noise_term

    return WTWt_v


def sym_toeplitz_derivative_quadratic_form(left_vector, right_vector):
    """
    Given a left vector v1 and a right vector v2, computes the quadratic form:
                                v1'*(dT/dc_i)*v2
    for all i, where dT/dc_i is the derivative of the Toeplitz matrix with respect to
    the ith element of its first column. Note that dT/dc_i is the same for any symmetric
    Toeplitz matrix T, so we do not require it as an argument.

    In particular, dT/dc_i is given by:
                                [0 0; I_{m-i+1} 0] + [0 I_{m-i+1}; 0 0]
    where I_{m-i+1} is the (m-i+1) dimensional identity matrix. In other words, dT/dc_i
    for i=1..m is the matrix with ones on the ith sub- and superdiagonal.

    Args:
        - left_vector (vector m) - left vector v1 in the quadratic form.
        - right_vector (vector m) - right vector v2 in the quadratic form.
    Returns:
        - vector m - a vector so that the ith element is the result of v1'*(dT/dc_i)*v2
    """
    m = len(left_vector)
    dT_dc_col = torch.zeros(m)

    dT_dc_row = left_vector
    dT_dc_col[0] = dT_dc_row[0]
    res = toeplitz_mv(dT_dc_col, dT_dc_row, right_vector)

    dT_dc_row = utils.reverse(left_vector)
    dT_dc_col[0] = dT_dc_row[0]
    res = res + toeplitz_mv(dT_dc_col, dT_dc_row, utils.reverse(right_vector))
    res[0] -= left_vector.dot(right_vector)

    return res
