#!/usr/bin/env python3

import torch
from torch.fft import fft, ifft

from ..utils import broadcasting


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

    toeplitz_shape = torch.Size((*toeplitz_column.shape, toeplitz_row.size(-1)))
    output_shape = broadcasting._matmul_broadcast_shape(toeplitz_shape, tensor.shape)
    broadcasted_t_shape = output_shape[:-1] if tensor.dim() > 1 else output_shape

    if tensor.ndimension() == 1:
        tensor = tensor.unsqueeze(-1)
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    tensor = tensor.expand(*output_shape)

    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0], toeplitz_row[0])
        )

    if type(toeplitz_column) != type(toeplitz_row) or type(toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    *batch_shape, orig_size, num_rhs = tensor.size()
    r_reverse = toeplitz_row[..., 1:].flip(dims=(-1,))

    c_r_rev = torch.zeros(*batch_shape, orig_size + r_reverse.size(-1), dtype=tensor.dtype, device=tensor.device)
    c_r_rev[..., :orig_size] = toeplitz_column
    c_r_rev[..., orig_size:] = r_reverse

    temp_tensor = torch.zeros(
        *batch_shape, 2 * orig_size - 1, num_rhs, dtype=toeplitz_column.dtype, device=toeplitz_column.device
    )
    temp_tensor[..., :orig_size, :] = tensor

    fft_M = fft(temp_tensor.transpose(-1, -2).contiguous())
    fft_c = fft(c_r_rev).unsqueeze(-2).expand_as(fft_M)
    fft_product = fft_M.mul_(fft_c)

    output = ifft(fft_product).real.transpose(-1, -2)
    output = output[..., :orig_size, :]
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
    r"""
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

    batch_shape = left_vectors.shape[:-2]
    toeplitz_size = left_vectors.size(-2)
    num_vectors = left_vectors.size(-1)

    left_vectors = left_vectors.transpose(-1, -2).contiguous()
    right_vectors = right_vectors.transpose(-1, -2).contiguous()

    columns = torch.zeros_like(left_vectors)
    columns[..., 0] = left_vectors[..., 0]
    res = toeplitz_matmul(columns, left_vectors, right_vectors.unsqueeze(-1))
    rows = left_vectors.flip(dims=(-1,))
    columns[..., 0] = rows[..., 0]
    res += toeplitz_matmul(columns, rows, torch.flip(right_vectors, dims=(-1,)).unsqueeze(-1))

    res = res.reshape(*batch_shape, num_vectors, toeplitz_size).sum(-2)
    res[..., 0] -= (left_vectors * right_vectors).view(*batch_shape, -1).sum(-1)

    return res
