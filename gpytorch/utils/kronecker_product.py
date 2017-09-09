import torch
from gpytorch.utils.toeplitz import sym_toeplitz_matmul, \
    sym_toeplitz_derivative_quadratic_form, index_coef_to_sparse
from torch.autograd import Variable


def kronecker_product(matrices):
    """
    Performs Kronecker product of a list of matrices.
    Args:
        - matrices (a list of matrices)
    Returns:
        - matrix - the result of kronecker_product of matrices
    """
    d = len(matrices)
    m = 1
    n = 1
    for i in range(d):
        m = m * matrices[i].size()[0]
        n = n * matrices[i].size()[1]

    if isinstance(matrices[0], Variable):
        res = Variable(torch.zeros(m, n))
    else:
        res = torch.zeros(m, n)

    for i in range(m):
        for j in range(n):
            if isinstance(matrices[0], Variable):
                old_v = Variable(torch.FloatTensor([1]))
            else:
                old_v = 1
            temp_m = m
            temp_n = n
            temp_i = i
            temp_j = j
            for k in range(d):
                temp_m = temp_m / matrices[k].size()[0]
                temp_n = temp_n / matrices[k].size()[1]
                new_v = old_v * matrices[k][int(temp_i / temp_m)][int(temp_j / temp_n)]
                temp_i = temp_i % temp_m
                temp_j = temp_j % temp_n
                old_v = new_v
            res[i, j] = old_v

    return res


def sym_kronecker_product_toeplitz_matmul(toeplitz_columns, tensor):
    return kronecker_product_toeplitz_matmul(toeplitz_columns, toeplitz_columns, tensor)


def kronecker_product_toeplitz_matmul(toeplitz_columns, toeplitz_rows, tensor):
    """
    Performs a tensor-vector multiplication Kv where the tensor K is T_0 \otimes \cdots \otimes T_{d-1}
    and T_i(i = 0, \cdots, d-1) are Toeplitz matrices.
    Args:
        - toeplitz_columns (d x m tensor) - columns of d toeplitz tensor T_i with
          length n_i
        - toelitz_rows (d x m tensor) - rows of toeplitz tensor T_i with length n_i
        - tensor (tensor n x p) - vector (p=1) or tensor (p>1) to multiply the
          Kronecker product of T_0, \cdots, T_{d-1} with, where n = n_0 * \cdots * n_{d-1}
    Returns:
        - tensor
    """
    output_dims = tensor.ndimension()
    if output_dims == 1:
        tensor = tensor.unsqueeze(1)

    if toeplitz_columns.ndimension() == 0:
        output = tensor

    else:
        n, p = tensor.size()

        d, n_0 = toeplitz_columns.size()

        if d == 1:
            output = sym_toeplitz_matmul(toeplitz_rows[0], tensor)

        else:
            len_sub = int(n / n_0)
            output = torch.zeros(n, p)

            for i in range(n_0):
                new_val = kronecker_product_toeplitz_matmul(toeplitz_columns[1:], toeplitz_rows[1:],
                                                            tensor[len_sub * i:len_sub * (i + 1)])
                output[len_sub * i:len_sub * (i + 1)] = new_val

            output = output.contiguous().view(n_0, len_sub * p)
            output = sym_toeplitz_matmul(toeplitz_rows[0], output)
            output = output.contiguous().view(n, p)
            output = output

    if output_dims == 1:
        output = output.squeeze(1)
    return output


def transpose_list_matrices(matrices):
    for i in range(len(matrices)):
        matrices[i] = (matrices[i].t())
    return matrices


def kp_interpolated_toeplitz_matmul(toeplitz_columns, tensor, interp_left=None, interp_right=None, noise_diag=None):
    """
    Given an interpolated matrix interp_left * T_1 \otimes ... \otimes T_d * interp_right, plus possibly an additional
    diagonal component s*I, compute a product with some tensor or matrix tensor, where T_i is
    symmetric Toeplitz matrices.

    Args:
        - toeplitz_columns (d x m matrix) - columns of d toeplitz matrix T_i with
          length n_i
        - interp_left (sparse matrix nxm) - Left interpolation matrix
        - interp_right (sparse matrix pxm) - Right interpolation matrix
        - tensor (matrix p x k) - Vector (k=1) or matrix (k>1) to multiply WKW with
        - noise_diag (tensor p) - If not none, add (s*I)tensor to WKW at the end.

    Returns:
        - tensor
    """
    output_dims = tensor.ndimension()
    noise_term = None

    if output_dims == 1:
        tensor = tensor.unsqueeze(1)

    if noise_diag is not None:
        noise_term = noise_diag.unsqueeze(1).expand_as(tensor) * tensor

    if interp_left is not None:
        # Get interp_{r}^{T} tensor
        interp_right_tensor = torch.dsmm(interp_right.t(), tensor)
        # Get (T interp_{r}^{T}) tensor
        rhs = kronecker_product_toeplitz_matmul(toeplitz_columns, toeplitz_columns, interp_right_tensor)

        # Get (interp_{l} T interp_{r}^{T})tensor
        output = torch.dsmm(interp_left, rhs)
    else:
        output = kronecker_product_toeplitz_matmul(toeplitz_columns, toeplitz_columns, tensor)

    if noise_term is not None:
        # Get (interp_{l} T interp_{r}^{T} + \sigma^{2}I)tensor
        output = output + noise_term

    if output_dims == 1:
        output = output.squeeze(1)
    return output


def kp_sym_toeplitz_derivative_quadratic_form(columns, left_vectors, right_vectors):
    """
    Given a left vector v1 and a right vector v2, computes the quadratic form:
                            v1'* (T_1x...xT_{i-1}x(dT_i/dc_i^j)xT_{i+1}x...T_d)*v2
    for all i and all j, where dT_i/dc_i^j is the derivative of the Toeplitz matrix with
    respect to the ith element of its first column.

    In particular, dT/dc^i is given by:
                                [0 0; I_{m-i+1} 0] + [0 I_{m-i+1}; 0 0]
    where I_{m-i+1} is the (m-i+1) dimensional identity matrix. In other words, dT/dc_i
    for i=1..m is the matrix with ones on the ith sub- and superdiagonal.

    Args:
        - toeplitz_columns (d x m matrix) - columns of d symmetric toeplitz matrix T_i
        - left_vector (vector m) - left vector v1 in the quadratic form.
        - right_vector (vector m) - right vector v2 in the quadratic form.
    Returns:
        - a list with length d of vectors - a list so that the jth element in the ith element of
          this list is the result of v1'*(dT_i/dc_i^j)*v2
    """

    if left_vectors.ndimension() == 1:
        left_vectors = left_vectors.unsqueeze(0)
        right_vectors = right_vectors.unsqueeze(0)

    res = torch.zeros(columns.size())
    d = columns.size()[0]
    s, m = left_vectors.size()

    for i in range(d):
        m_left = 1
        for j in range(i):
            m_left = m_left * len(columns[j])
        m_right = 1
        for j in range(i + 1, d):
            m_right = m_right * len(columns[j])

        m_i = len(columns[i])

        right_vectors_i = right_vectors.clone()
        right_vectors_i = right_vectors_i.view(s, m_left, m_right * m_i)

        left_vectors_i = left_vectors.clone()
        left_vectors_i = left_vectors_i.view(s, int(m / m_left), m_left)

        if i == 0:
            left_vectors_i = left_vectors_i.transpose(1, 2)
        else:
            left_vectors_i = left_vectors_i.transpose(1, 2)
            left_vectors_i = left_vectors_i.transpose(0, 1).contiguous().view(m_left, s * int(m / m_left))
            left_vectors_i = kronecker_product_toeplitz_matmul(columns[:i], columns[:i], left_vectors_i)
            left_vectors_i = left_vectors_i.contiguous().view(m_left, s, int(m / m_left)).transpose(0, 1)

        for j in range(m_left):
            right_vectors_i_j = right_vectors_i[:, j].contiguous().view(s, m_i, m_right).transpose(1, 2)

            left_vectors_i_j = left_vectors_i[:, j].contiguous()
            left_vectors_i_j = left_vectors_i_j.view(s, m_i, m_right)
            if i == d - 1:
                left_vectors_i_j = left_vectors_i_j.transpose(1, 2)
            else:
                left_vectors_i_j = left_vectors_i_j.transpose(1, 2)
                left_vectors_i_j = left_vectors_i_j.transpose(0, 1).contiguous().view(m_i, s * m_right)
                left_vectors_i_j = kronecker_product_toeplitz_matmul(columns[i + 1:], columns[i + 1:],
                                                                     left_vectors_i_j)
                left_vectors_i_j = left_vectors_i_j.contiguous().view(m_i, s, m_right).transpose(0, 1)
            for k in range(m_right):
                res[i] = res[i] + sym_toeplitz_derivative_quadratic_form(left_vectors_i_j[:, k],
                                                                         right_vectors_i_j[:, k])
    return res


def list_of_indices_and_values_to_sparse(index_matrices, value_matrices, columns):
    index_matrix, value_matrix, m = _merge_index_and_value_matrices(index_matrices, value_matrices, columns)
    return index_coef_to_sparse(index_matrix, value_matrix, m)


def _merge_index_and_value_matrices(index_matrices, value_matrices, columns):
    d = len(index_matrices)
    if d == 1:
        return index_matrices[0], value_matrices[0], len(columns[0])

    index_matrices_1, value_matrices_1, m_1 = _merge_index_and_value_matrices(index_matrices[1:],
                                                                              value_matrices[1:],
                                                                              columns[1:])
    index_matrix_0, value_matrix_0 = index_matrices[0], value_matrices[0]

    n = index_matrix_0.size()[0]

    index_matrix = torch.zeros(n, index_matrices_1.size()[1] * index_matrix_0.size()[1])
    value_matrix = torch.zeros(n, value_matrices_1.size()[1] * value_matrix_0.size()[1])

    m_0 = len(columns[0])

    for i in range(n):
        index = 0
        for j in range(index_matrix_0.size()[1]):
            for k in range(index_matrices_1.size()[1]):
                index_matrix[i][index] = m_1 * index_matrix_0[i][j] + index_matrices_1[i][k]
                value_matrix[i][index] = value_matrix_0[i][j] * value_matrices_1[i][k]
                index = index + 1

    return index_matrix.long(), value_matrix, m_0 * m_1
