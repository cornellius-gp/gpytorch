#!/usr/bin/env python3

import torch
from ..utils import fft


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
        raise RuntimeError("toeplitz_column must be a vector.")

    if toeplitz_row.ndimension() != 1:
        raise RuntimeError("toeplitz_row must be a vector.")

    if toeplitz_column[0] != toeplitz_row[0]:
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if len(toeplitz_column) != len(toeplitz_row):
        raise RuntimeError("c and r should have the same length " "(Toeplitz matrices are necessarily square).")

    if type(toeplitz_column) != type(toeplitz_row):
        raise RuntimeError("toeplitz_column and toeplitz_row should be the same type.")

    if len(toeplitz_column) == 1:
        return toeplitz_column.view(1, 1)

    res = torch.empty(
        len(toeplitz_column), len(toeplitz_column), dtype=toeplitz_column.dtype, device=toeplitz_column.device
    )
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
        raise RuntimeError("c and r should have the same length (Toeplitz matrices are necessarily square).")

    is_batch = True
    if toeplitz_column.ndimension() == 1:
        toeplitz_column = toeplitz_column.unsqueeze(0)
        toeplitz_row = toeplitz_row.unsqueeze(0)
        if tensor.ndimension() < 3:
            tensor = tensor.unsqueeze(0)
        is_batch = False

    if toeplitz_column.ndimension() != 2:
        raise RuntimeError(
            "The first two inputs to ToeplitzMV should be vectors \
                            (or matrices, representing batch) \
                            (first column c and row r of the Toeplitz matrix)"
        )

    if toeplitz_column.size(0) == 1:
        toeplitz_column = toeplitz_column.expand(tensor.size(0), toeplitz_column.size(1))
        toeplitz_row = toeplitz_row.expand(tensor.size(0), toeplitz_row.size(1))

    if toeplitz_column.size()[:2] != tensor.size()[:2]:
        raise RuntimeError(
            "Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix "
            "against a matrix with leading dimension {}.".format(
                len(toeplitz_column), len(toeplitz_column), len(tensor)
            )
        )

    if not torch.equal(toeplitz_column[:, 0], toeplitz_row[:, 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    output_dims = tensor.ndimension()
    if output_dims == 2:
        tensor = tensor.unsqueeze(2)

    if toeplitz_column.size(1) == 1:
        output = toeplitz_column.view(-1, 1, 1).matmul(tensor)

    else:
        batch_size, orig_size, num_rhs = tensor.size()
        r_reverse = toeplitz_row[:, 1:].flip(dims=(1,))

        c_r_rev = torch.zeros(batch_size, orig_size + r_reverse.size(1), dtype=tensor.dtype, device=tensor.device)
        c_r_rev[:, :orig_size] = toeplitz_column
        c_r_rev[:, orig_size:] = r_reverse

        temp_tensor = torch.zeros(
            batch_size, 2 * orig_size - 1, num_rhs, dtype=toeplitz_column.dtype, device=toeplitz_column.device
        )
        temp_tensor[:, :orig_size, :] = tensor

        fft_M = fft.fft1(temp_tensor.transpose(1, 2).contiguous())
        fft_c = fft.fft1(c_r_rev).unsqueeze(1).expand_as(fft_M)
        fft_product = torch.zeros_like(fft_M)

        fft_product[:, :, :, 0].addcmul_(fft_c[:, :, :, 0], fft_M[:, :, :, 0])
        fft_product[:, :, :, 0].addcmul_(-1, fft_c[:, :, :, 1], fft_M[:, :, :, 1])
        fft_product[:, :, :, 1].addcmul_(fft_c[:, :, :, 1], fft_M[:, :, :, 0])
        fft_product[:, :, :, 1].addcmul_(fft_c[:, :, :, 0], fft_M[:, :, :, 1])

        output = fft.ifft1(fft_product).transpose(1, 2)
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
        left_vectors = left_vectors.unsqueeze(1)
        right_vectors = right_vectors.unsqueeze(1)

    left_vectors = left_vectors.transpose(-1, -2)
    right_vectors = right_vectors.transpose(-1, -2)

    if left_vectors.ndimension() == 3:
        batch = True
        batch_size, s, _ = left_vectors.size()
        left_vectors = left_vectors.contiguous().view(batch_size * s, -1)
        right_vectors = right_vectors.contiguous().view(batch_size * s, -1)
    else:
        batch = False

    s, m = left_vectors.size()

    left_vectors.contiguous()
    right_vectors.contiguous()

    columns = torch.zeros(s, m, dtype=left_vectors.dtype, device=left_vectors.device)
    columns[:, 0] = left_vectors[:, 0]
    res = toeplitz_matmul(columns, left_vectors, right_vectors)
    rows = left_vectors.flip(dims=(1,))
    columns[:, 0] = rows[:, 0]
    res += toeplitz_matmul(columns, rows, torch.flip(right_vectors, dims=(1,)))

    if not batch:
        res = res.sum(0)
        res[0] -= (left_vectors * right_vectors).sum()
    else:
        res = res.contiguous().view(batch_size, -1, m).sum(1)
        res[:, 0] -= (left_vectors * right_vectors).view(batch_size, -1).sum(1)

    return res
