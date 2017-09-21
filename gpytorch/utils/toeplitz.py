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
        - (Sparse matrix n-by-row_length) - A torch sparse matrix W so that
            W[i, index_matrix[i, j]] = value_matrix[i, j].
    """
    num_target_points, num_coefficients = value_matrix.size()

    row_tensor = torch.arange(0, num_target_points).unsqueeze(1)
    row_tensor = row_tensor.repeat(1, num_coefficients).type_as(index_matrix)
    index_tensor = torch.cat([row_tensor.view(1, -1), index_matrix.view(1, -1)], 0)
    value_tensor = value_matrix.view(-1)

    nonzero_indices = value_tensor.nonzero()
    if nonzero_indices.storage():
        nonzero_indices.squeeze_()
        index_tensor = index_tensor.index_select(1, nonzero_indices)
        value_tensor = value_tensor.index_select(0, nonzero_indices)
    else:
        index_tensor = index_tensor.resize_(2, 1).zero_()
        value_tensor = value_tensor.resize_(1).zero_()

    res = torch.sparse.FloatTensor(index_tensor, value_tensor, torch.Size([num_target_points, row_length]))
    return res


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


def toeplitz_matmul(toeplitz_column, toeplitz_row, tensor):
    """
    Performs multiplication T * M where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - tensor (matrix n x p or b x n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor (n x p or b x n x p) - The result of the matrix multiply T * M.
    """
    if toeplitz_column.size() != toeplitz_row.size():
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    is_batch = True
    if toeplitz_column.ndimension() == 1:
        toeplitz_column = toeplitz_column.unsqueeze(0)
        toeplitz_row = toeplitz_row.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        is_batch = False

    if toeplitz_column.ndimension() != 2:
        raise RuntimeError('The first two inputs to ToeplitzMV should be vectors \
                            (or matrices, representing batch) \
                            (first column c and row r of the Toeplitz matrix)')

    if toeplitz_column.size()[:2] != tensor.size()[:2]:
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a matrix with \
                            leading dimension {}.'.format(len(toeplitz_column), len(toeplitz_column), len(tensor)))

    if not torch.equal(toeplitz_column[:, 0], toeplitz_row[:, 0]):
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first element, \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(toeplitz_column[0], toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(tensor):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    output_dims = tensor.ndimension()
    if output_dims == 2:
        tensor = tensor.unsqueeze(2)

    if toeplitz_column.size(1) == 1:
        output = toeplitz_column.view(-1, 1, 1).matmul(tensor)

    else:
        batch_size, orig_size, num_rhs = tensor.size()
        r_reverse = utils.reverse(toeplitz_row[:, 1:], dim=1)

        c_r_rev = toeplitz_column.new(batch_size, orig_size + r_reverse.size(1)).zero_()
        c_r_rev[:, :orig_size] = toeplitz_column
        c_r_rev[:, orig_size:] = r_reverse

        temp_tensor = toeplitz_column.new(batch_size, 2 * orig_size - 1, num_rhs).zero_()
        temp_tensor[:, :orig_size, :] = tensor

        fft_M = fft.fft1(temp_tensor.transpose(1, 2).contiguous())
        fft_c = fft.fft1(c_r_rev).unsqueeze(1).expand_as(fft_M)
        fft_product = toeplitz_column.new(fft_M.size()).zero_()

        fft_product[:, :, :, 0].addcmul_(fft_c[:, :, :, 0], fft_M[:, :, :, 0])
        fft_product[:, :, :, 0].addcmul_(-1, fft_c[:, :, :, 1], fft_M[:, :, :, 1])
        fft_product[:, :, :, 1].addcmul_(fft_c[:, :, :, 1], fft_M[:, :, :, 0])
        fft_product[:, :, :, 1].addcmul_(fft_c[:, :, :, 0], fft_M[:, :, :, 1])

        output = fft.ifft1(fft_product, (batch_size, num_rhs, 2 * orig_size - 1)).transpose(1, 2)
        output = output[:, :orig_size, :]

    if output_dims == 2:
        output = output.squeeze(2)

    if not is_batch:
        output = output.squeeze(0)

    return output


def sym_toeplitz_matmul(toeplitz_column, tensor):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor
    """
    return toeplitz_matmul(toeplitz_column, toeplitz_column, tensor)


def interpolated_sym_toeplitz_matmul(toeplitz_column, vector, W_left=None, W_right=None, noise_diag=None):
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
        - matrix nxk
    """
    noise_term = None
    ndim = vector.ndimension()

    if ndim == 1:
        vector = vector.unsqueeze(1)

    if noise_diag is not None:
        noise_term = noise_diag.unsqueeze(1).expand_as(vector) * vector

    if W_left is not None:
        # Get W_{r}^{T}vector
        Wt_times_v = torch.dsmm(W_right.t(), vector)
        # Get (TW_{r}^{T})vector
        TWt_v = sym_toeplitz_matmul(toeplitz_column, Wt_times_v)

        # Get (W_{l}TW_{r}^{T})vector
        WTWt_v = torch.dsmm(W_left, TWt_v)
    else:
        WTWt_v = sym_toeplitz_matmul(toeplitz_column, vector)

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)vector
        WTWt_v = WTWt_v + noise_term

    if ndim == 1:
        WTWt_v = WTWt_v.squeeze(1)

    return WTWt_v


def sym_toeplitz_derivative_quadratic_form(left_vectors, right_vectors):
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
        - left_vectors (vector m or matrix s x m) - s left vectors u[j] in the quadratic form.
        - right_vectors (vector m or matrix s x m) - s right vectors v[j] in the quadratic form.
    Returns:
        - vector m - a vector so that the ith element is the result of \sum_j(u[j]*(dT/dc_i)*v[j])
    """
    if left_vectors.ndimension() == 1:
        left_vectors = left_vectors.unsqueeze(0)
        right_vectors = right_vectors.unsqueeze(0)
    s, m = left_vectors.size()
    dT_dc_col = torch.zeros(m)

    res = torch.zeros(m)

    left_vectors.contiguous()
    right_vectors.contiguous()
    for j in range(s):
        dT_dc_row = left_vectors[j]
        dT_dc_col[0] = dT_dc_row[0]
        res += toeplitz_matmul(dT_dc_col, dT_dc_row, right_vectors[j])
        dT_dc_row = utils.reverse(left_vectors[j])
        dT_dc_col[0] = dT_dc_row[0]
        res = res + toeplitz_matmul(dT_dc_col, dT_dc_row, utils.reverse(right_vectors[j]))
    res[0] -= (left_vectors * right_vectors).sum()

    return res
