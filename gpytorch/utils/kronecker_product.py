import torch
from gpytorch.utils.toeplitz import toeplitz_mv, toeplitz_mm, \
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


def sym_kronecker_product_toeplitz_mul(toeplitz_columns, vector):
    return kronecker_product_toeplitz_mul(toeplitz_columns, toeplitz_columns, vector)


def kronecker_product_toeplitz_mul(toeplitz_columns, toeplitz_rows, vector):
    """
    Performs a matrix-vector multiplication Kv where the matrix K is T_0 \otimes \cdots \otimes T_{d-1}
    and T_i(i = 0, \cdots, d-1) are Toeplitz matrices.
    Args:
        - toeplitz_columns (d x m matrix) - columns of d toeplitz matrix T_i with
          length n_i
        - toelitz_rows (d x m matrix) - rows of toeplitz matrix T_i with length n_i
        - vector (matrix n x p) - vector (p=1) or matrix (p>1) to multiply the
          Kronecker product of T_0, \cdots, T_{d-1} with, where n = n_0 * \cdots * n_{d-1}
    Returns:
        - vector n or matrix n x p
    """
    if toeplitz_columns.ndimension() == 0:
        return vector
    if vector.ndimension() == 1:
        mul_func = toeplitz_mv
        n = len(vector)
        p = 1
    else:
        mul_func = toeplitz_mm
        n, p = vector.size()

    d, n_0 = toeplitz_columns.size()

    if d == 1:
        return mul_func(toeplitz_columns[0], toeplitz_rows[0], vector)

    len_sub = int(n / n_0)
    res = torch.zeros(n, p)

    for i in range(n_0):
        res[len_sub * i:len_sub * (i + 1)] = kronecker_product_toeplitz_mul(toeplitz_columns[1:],
                                                                            toeplitz_rows[1:],
                                                                            vector[len_sub * i:len_sub * (i + 1)])
    res = res.contiguous().view(n_0, len_sub * p)

    # res = kronecker_product_toeplitz_mul(toeplitz_columns[1:], toeplitz_rows[1:], vector.view(n_0, len_sub).t())
    # res = res.t()

    res = toeplitz_mm(toeplitz_columns[0], toeplitz_rows[0], res)
    res = res.contiguous().view(n, p)
    res = res.squeeze()
    return res


def kronecker_product_mul(matrices, vector):
    """
    Performs a matrix-vector multiplication Kv or a matrix-matrix multiplication KM, where K = K_1 \otimes
    \cdots \otimes K_d and K_i(i = 1, \cdots, d) are normal matrices or sparse matrices.
    Args:
        - matrices (a list with length d of matrices) - d matrices(m_i x n_i) in Kronecker Product
        - vector (Matrix n x p) - vector (p=1) or matrix (p>1) to multiply the Kronecker product
                                    of K_1, \cdots, K_d with, where n = n_0 * \cdots * n_{d-1}
    Returns:
        - vector m or matrix m x p - m = m_0 * \cdots * m_{d-1}
    """
    if len(matrices) == 0:
        return vector
    if vector.ndimension() == 1:
        mul_func = torch.mv
        n = vector.size()
        p = 1
    else:
        mul_func = torch.mm
        n, p = vector.size()
    d = len(matrices)
    m_0, n_0 = matrices[0].size()
    m = 1
    for i in range(d):
        m = m * len(matrices[i])

    if d == 1:
        return mul_func(matrices[0], vector)

    len_sub_v = int(n / n_0)
    len_sub_res = int(m / m_0)
    res = torch.zeros(n_0 * len_sub_res, p)

    for i in range(n_0):
        res[len_sub_res * i:len_sub_res * (i + 1)] = kronecker_product_mul(matrices[1:],
                                                                           vector[len_sub_v * i:len_sub_v * (i + 1)])

    res = res.contiguous().view(n_0, len_sub_res * p)
    res = mul_func(matrices[0], res)
    res = res.contiguous().view(m, p)

    res = res.squeeze()
    return res


def transpose_list_matrices(matrices):
    for i in range(len(matrices)):
        matrices[i] = (matrices[i].t())
    return matrices


def kp_interpolated_toeplitz_mul(toeplitz_columns, vector, W_left=None, W_right=None, noise_diag=None):
    """
    Given an interpolated matrix W_left * T_1 \otimes ... \otimes T_d * W_right, plus possibly an additional
    diagonal component s*I, compute a product with some vector or matrix vector, where T_i is
    symmetric Toeplitz matrices.

    Args:
        - toeplitz_columns (d x m matrix) - columns of d toeplitz matrix T_i with
          length n_i
        - W_left (sparse matrix nxm) - Left interpolation matrix
        - W_right (sparse matrix pxm) - Right interpolation matrix
        - vector (matrix p x k) - Vector (k=1) or matrix (k>1) to multiply WKW with
        - noise_diag (vector p) - If not none, add (s*I)vector to WKW at the end.

    Returns:
        - vector p or matrix p x k - The result of multiplying (K + sI)vector if noise_diag exists, or (WTW)vector
                                    otherwise, where K = \otimes_{i=1}^d W_left_i * T_i * W_right_i .
    """
    noise_term = None
    if vector.ndimension() == 1:
        if noise_diag is not None:
            noise_term = noise_diag.expand_as(vector) * vector
        vector = vector.unsqueeze(1)
    else:
        if noise_diag is not None:
            noise_term = noise_diag.unsqueeze(1).expand_as(vector) * vector

    if W_left is not None:
        # Get W_{r}^{T}vector
        Wt_times_v = torch.dsmm(W_right.t(), vector)
        # Get (TW_{r}^{T})vector
        KWt_v = kronecker_product_toeplitz_mul(toeplitz_columns, toeplitz_columns, Wt_times_v.squeeze())

        if KWt_v.ndimension() == 1:
            KWt_v.unsqueeze_(1)

        # Get (W_{l}TW_{r}^{T})vector
        WKWt_v = torch.dsmm(W_left, KWt_v).squeeze()
    else:
        WKWt_v = kronecker_product_toeplitz_mul(toeplitz_columns, toeplitz_columns, vector)

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)vector
        WKWt_v = WKWt_v + noise_term

    return WKWt_v


def kp_sym_toeplitz_derivative_quadratic_form(columns, left_vector, right_vector):
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

    res = torch.zeros(columns.size())
    d = columns.size()[0]
    m = len(left_vector)
    for i in range(d):
        m_left = 1
        for j in range(i):
            m_left = m_left * len(columns[j])
        m_right = 1
        for j in range(i + 1, d):
            m_right = m_right * len(columns[j])

        m_i = len(columns[i])

        right_vector_i = right_vector.clone()
        right_vector_i = right_vector_i.view(m_left, m_right * m_i)

        left_vector_i = left_vector.clone()
        left_vector_i = left_vector_i.view(int(m / m_left), m_left)

        if i == 0:
            left_vector_i = left_vector_i.t()
        else:
            left_vector_i = kronecker_product_toeplitz_mul(columns[:i], columns[:i], left_vector_i.t())

        for j in range(m_left):
            right_vector_i_j = right_vector_i[j].view(m_i, m_right).t()

            left_vector_i_j = left_vector_i[j]
            left_vector_i_j = left_vector_i_j.contiguous().view(m_i, m_right)
            if i == d - 1:
                left_vector_i_j = left_vector_i_j.t()
            else:
                left_vector_i_j = kronecker_product_toeplitz_mul(columns[i + 1:], columns[i + 1:], left_vector_i_j.t())
            for k in range(m_right):
                res[i] = res[i] + sym_toeplitz_derivative_quadratic_form(left_vector_i_j[k], right_vector_i_j[k])
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
