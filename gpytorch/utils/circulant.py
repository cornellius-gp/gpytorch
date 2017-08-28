import torch
from gpytorch import utils
from gpytorch.utils import fft


def _get_rotation_indices(n, rotation):
    if rotation > 0:
        rotate_index = torch.LongTensor(n)
        rotate_index[:rotation] = torch.LongTensor(list(range(n - rotation, n)))
        rotate_index[rotation:] = torch.LongTensor(list(range(0, n - rotation)))
    elif rotation < 0:
        rotation = -rotation
        rotate_index = torch.LongTensor(n)
        rotate_index[:-rotation] = torch.LongTensor(list(range(rotation, n)))
        rotate_index[-rotation:] = torch.LongTensor(list(range(0, rotation)))
    else:
        rotate_index = torch.LongTensor(list(range(n)))

    return rotate_index


def rotate(input, rotation=1, dim=0):
    n = input.size(dim)
    rotate_index = _get_rotation_indices(n, rotation)
    return input.index_select(dim, rotate_index)


def circulant(circulant_column):
    n = len(circulant_column)
    A = circulant_column.new().resize_(n, n).fill_(0)

    A[:, 0] = circulant_column
    for i in range(1, n):
        A[:, i] = rotate(circulant_column, i)

    return A


def left_rotate_trace(input, rotation=1):
    n = input.size(0)
    rotate_index = _get_rotation_indices(n, rotation)
    column_index = torch.LongTensor(list(range(n)))

    trace_value = input[rotate_index, column_index].sum()

    return trace_value


def right_rotate_trace(input, rotation=1):
    n = input.size(0)
    rotate_index = _get_rotation_indices(n, rotation)
    row_index = torch.LongTensor(list(range(n)))

    trace_value = input[row_index, rotate_index].sum()

    return trace_value


def circulant_transpose(input_column):
    output_column = input_column.new().resize_as_(input_column)
    output_column[0] = input_column[0]
    output_column[1:] = utils.reverse(input_column[1:])

    return output_column


def circulant_mv(circulant_column, vector):
    """
    Performs a matrix-vector multiplication Cv where the matrix C is circulant.
    Args:
        - circulant_column (vector n) - First column of the circulant matrix C.
        - vector (vector n) - Vector to multiply the circulant matrix with.
    Returns:
        - Vector (n) - The result of the matrix-vector multiply Cv.
    """
    if circulant_column.ndimension() != 1 or vector.ndimension() != 1:
        raise RuntimeError('All inputs to CirculantMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector vector).')

    if len(circulant_column) != len(vector):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if type(circulant_column) != type(vector):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    if len(circulant_column) == 1:
        return (circulant_column.view(1, 1).mv(vector))

    fft_c = fft.fft1(circulant_column)
    fft_v = fft.fft1(vector)

    fft_product = torch.zeros(fft_c.size())

    fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
    fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
    fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
    fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

    res = fft.ifft1(fft_product, vector.size())

    return res


def circulant_mm(circulant_column, matrix):
    """
    Performs a matrix-matrix multiplication CM where the matrix C is circulant.
    Args:
        - circulant_column (vector n) - First column of the circulant matrix C.
        - matrix (matrix n x p) - Matrix to multiply the Toeplitz matrix with.
    Returns:
        - Matrix (n x p) - The result of the matrix-vector multiply CM.
    """
    if circulant_column.ndimension() != 1 or matrix.ndimension() != 2:
        raise RuntimeError('All inputs to CirculantMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector vector).')

    if len(circulant_column) != len(matrix):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if type(circulant_column) != type(matrix):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    if len(circulant_column) == 1:
        return (circulant_column.view(1, 1).mv(matrix))

    fft_M = fft.fft1(matrix.t().contiguous())
    fft_c = fft.fft1(circulant_column).expand_as(fft_M)
    fft_product = torch.zeros(fft_M.size())

    fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
    fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

    res = fft.ifft1(fft_product, matrix.size()).t()

    return res


def circulant_invmm(circulant_column, matrix):
    """
    Performs a batch of linear solves C^{-1}M where the matrix C is circulant.
    Args:
        - circulant_column (vector n) - First column of the circulant matrix C.
        - matrix (matrix n x p) - Matrix to multiply the Toeplitz matrix with.
    Returns:
        - Matrix (n x p) - The result of the linear solves C^{-1}M.
    """
    if circulant_column.ndimension() != 1 or matrix.ndimension() != 2:
        raise RuntimeError('All inputs to CirculantMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector vector).')

    if len(circulant_column) != len(matrix):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if type(circulant_column) != type(matrix):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    if len(circulant_column) == 1:
        return (circulant_column.view(1, 1).mv(matrix))

    fft_M = fft.fft1(matrix.t().contiguous())
    fft_c = fft.fft1(circulant_column)

    denominator = fft_c[:, 0].pow(2) + fft_c[:, 1].pow(2)
    fft_c[:, 0] = fft_c[:, 0] / denominator
    fft_c[:, 1] = -fft_c[:, 1] / denominator
    fft_c = fft_c.expand_as(fft_M)

    fft_product = torch.zeros(fft_M.size())

    fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
    fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

    res = fft.ifft1(fft_product, matrix.size()).t()

    return res


def circulant_invmv(circulant_column, vector):
    """
    Performs a linear solve C^{-1}v where the matrix C is circulant.
    Args:
        - circulant_column (vector n) - First column of the circulant matrix C.
        - vector (vector n) - Vector to multiply the circulant matrix with.
    Returns:
        - Vector (n) - The result of the linear solve C^{-1}v.
    """
    if circulant_column.ndimension() != 1 or vector.ndimension() != 1:
        raise RuntimeError('All inputs to CirculantMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector vector).')

    if len(circulant_column) != len(vector):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if type(circulant_column) != type(vector):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    if len(circulant_column) == 1:
        return (circulant_column.view(1, 1).mv(vector))

    fft_c = fft.fft1(circulant_column)
    denominator = fft_c[:, 0].pow(2) + fft_c[:, 1].pow(2)
    fft_c[:, 0] = fft_c[:, 0] / denominator
    fft_c[:, 1] = -fft_c[:, 1] / denominator

    fft_v = fft.fft1(vector)

    fft_product = torch.zeros(fft_c.size())
    fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
    fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
    fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
    fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

    res = fft.ifft1(fft_product, vector.size())

    return res


def frobenius_circulant_approximation(A):
    n = A.size(0)
    F = A.new().resize_(n).fill_(0)

    for r in range(n):
        F[r] = right_rotate_trace(A.t(), -r) + left_rotate_trace(A, r)

    F = F / (2 * n)

    return F


def frobenius_circulant_approximation_toeplitz(toeplitz_column):
    n = toeplitz_column.size(0)
    circulant_column = toeplitz_column.new().resize_(n).fill_(0)

    for i in range(n):
        circulant_column[i] = (i * toeplitz_column[-i] + (n - i) * toeplitz_column[i]) / n

    circulant_column[0] += 1e-4
    return circulant_column
