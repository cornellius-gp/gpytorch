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


def toeplitz(c, r):
    """
    Constructs tensor version of toeplitz matrix from column vector
    Args:
        - c (vector n) - column of toeplitz matrix
        - r (vector n-1) - row of toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    assert c.ndimension() == 1
    assert r.ndimension() == 1
    assert c[0] == r[0]
    assert len(c) == len(r)
    assert type(c) == type(r)

    res = torch.Tensor(len(c), len(c)).type_as(c)
    for i, val in enumerate(c):
        for j in range(len(c) - i):
            res[j + i, j] = val
    for i, val in list(enumerate(r))[1:]:
        for j in range(len(r) - i):
            res[j, j + i] = val
    return res


def sym_toeplitz(c):
    """
    Constructs tensor version of symmetric toeplitz matrix from column vector
    Args:
        - c (vector n) - column of Toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    return toeplitz(c, c)


def toeplitz_getitem(c, r, i, j):
    """
    Gets the (i,j)th entry of a Toeplitz matrix T.
    Args:
        - c (vector n) - column of Toeplitz matrix
        - r (vector n) - row of Toeplitz matrix
        - i (scalar) - row of entry to get
        - j (scalar) - column of entry to get
    Returns:
        - T[i,j], where T is the Toeplitz matrix specified by c and r.
    """
    index = i - j
    if index < 0:
        return r[abs(index)]
    else:
        return c[index]


def toeplitz_mm(c, r, M):
    if c.ndimension() != 1 or r.ndimension() != 1 or M.ndimension() != 2:
        raise RuntimeError('The first two inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                            matrix), and the last input should be a matrix.')

    if len(c) != len(r):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if len(c) != len(M):
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a matrix with leading \
                            dimension {}.'.format(len(c), len(c), len(M)))

    if c[0] != r[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first element, \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(c[0], r[0]))

    if type(c) != type(r) or type(c) != type(M):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    _, num_rhs = M.size()
    orig_size = len(c)
    r_reverse = utils.reverse(r[1:])
    c.resize_(orig_size + len(r_reverse))
    c[orig_size:].copy_(r_reverse)

    M.resize_(2 * orig_size - 1, num_rhs)
    M[orig_size:, :].fill_(0)

    fft_M = fft.fft1(M.t().contiguous())
    fft_c = fft.fft1(c).expand_as(fft_M)
    fft_product = torch.zeros(fft_M.size())

    fft_product[:, :, 0].addcmul_(fft_c[:, :, 0], fft_M[:, :, 0])
    fft_product[:, :, 0].addcmul_(-1, fft_c[:, :, 1], fft_M[:, :, 1])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 1], fft_M[:, :, 0])
    fft_product[:, :, 1].addcmul_(fft_c[:, :, 0], fft_M[:, :, 1])

    res = fft.ifft1(fft_product, (num_rhs, 2 * orig_size - 1)).t()
    c.resize_(orig_size)
    r.resize_(orig_size)
    M.resize_(orig_size, num_rhs)
    res = res[:orig_size, :]
    return res


def toeplitz_mv(c, r, v):
    if c.ndimension() != 1 or r.ndimension() != 1 or v.ndimension() != 1:
        raise RuntimeError('All inputs to ToeplitzMV should be vectors (first column c and row r of the Toeplitz \
                            matrix plus the target vector v).')

    if len(c) != len(r):
        raise RuntimeError('c and r should have the same length (Toeplitz matrices are necessarily square).')

    if len(c) != len(v):
        raise RuntimeError('Dimension mismatch: attempting to multiply a {}x{} Toeplitz matrix against a length \
                            {} vector.'.format(len(c), len(c), len(v)))

    if c[0] != r[0]:
        raise RuntimeError('The first column and first row of the Toeplitz matrix should have the same first \
                            otherwise the value of T[0,0] is ambiguous. \
                            Got: c[0]={} and r[0]={}'.format(c[0], r[0]))

    if type(c) != type(r) or type(c) != type(v):
        raise RuntimeError('The types of all inputs to ToeplitzMV must match.')

    orig_size = len(c)
    r_reverse = utils.reverse(r[1:])
    c.resize_(orig_size + len(r_reverse))
    c[orig_size:].copy_(r_reverse)

    v.resize_(2 * orig_size - 1)
    v[orig_size:].fill_(0)

    fft_c = fft.fft1(c)
    fft_v = fft.fft1(v)
    fft_product = torch.zeros(fft_c.size())

    fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
    fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
    fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
    fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

    res = fft.ifft1(fft_product, c.size())
    c.resize_(orig_size)
    r.resize_(orig_size)
    v.resize_(orig_size)
    res.resize_(orig_size)
    return res


def interpolated_toeplitz_mul(c, v, W_left=None, W_right=None, noise_diag=None):
    """
    Given a interpolated symmetric Toeplitz matrix W_left*T*W_right, plus possibly an additional
    diagonal component s*I, compute a matrix-vector product with some vector or matrix v.

    Args:
        - c (vector m) - First column of the symmetric Toeplitz matrix T
        - W_left (sparse matrix nxm) - Left interpolation matrix
        - W_right (sparse matrix pxm) - Right interpolation matrix
        - v (matrix pxk) - Vector (k=1) or matrix (k>1) to multiply WTW with
        - noise_diag (vector p) - If not none, add (s*I)v to WTW at the end.

    Returns:
        - matrix nxk - The result of multiplying (WTW + sI)v if noise_diag exists, or (WTW)v otherwise.
    """
    noise_term = None
    if v.ndimension() == 1:
        if noise_diag is not None:
            noise_term = noise_diag.expand_as(v) * v
        v = v.unsqueeze(1)
        mul_func = utils.toeplitz.toeplitz_mv
    else:
        if noise_diag is not None:
            noise_term = noise_diag.unsqueeze(1).expand_as(v) * v
        mul_func = utils.toeplitz.toeplitz_mm

    if W_left is not None:
        # Get W_{r}^{T}v
        Wt_times_v = torch.dsmm(W_right.t(), v)
        # Get (TW_{r}^{T})v
        TWt_v = mul_func(c, c, Wt_times_v.squeeze())

        if TWt_v.ndimension() == 1:
            TWt_v.unsqueeze_(1)

        # Get (W_{l}TW_{r}^{T})v
        WTWt_v = torch.dsmm(W_left, TWt_v).squeeze()
    else:
        WTWt_v = mul_func(c, c, v)

    if noise_term is not None:
        # Get (W_{l}TW_{r}^{T} + \sigma^{2}I)v
        WTWt_v = WTWt_v + noise_term

    return WTWt_v
