import logging
from .interpolation import Interpolation
from .lbfgs import LBFGS
from .lincg import LinearCG
from .slq_logdet import SLQLogDet


__all__ = [
    Interpolation,
    LBFGS,
    LinearCG,
    SLQLogDet,
]


class pd_catcher(object):
    '''
    A decorator to deal with non-positive definite matrices (useful during optimization)
    If an error due to non-psotiive definiteness occurs when calling the function, we
    retry the function call a certain number of times.
    After a certain number of trials, it fails.
    '''
    def __init__(self, catch_function=None, max_trials=20, log_interval=5):
        self.catch_function = catch_function
        self.n_trials = 0
        self.max_trials = max_trials
        self.log_interval = log_interval

    def __call__(self, function):
        def wrapped_function(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
                self.n_trials = 0
            except (ZeroDivisionError, RuntimeError) as e:
                if self.n_trials < self.max_trials:
                    if self.catch_function:
                        result = self.catch_function(*args, **kwargs)
                    self.n_trials += 1
                    if self.n_trials % self.log_interval == 0:
                        logging.warning('Not PD matrix: %d more attempts' % (self.max_trials - self.n_trials))
                else:
                    raise e
            return result
        return wrapped_function


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


def sym_toeplitz(c):
    """
    Constructs tensor version of symmetric toeplitz matrix from column vector
    Args:
        - c (vector n) - column of Toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
    """
    return toeplitz(c, c)


def reverse(input, dim=0):
    """
    Reverses a tensor
    Args:
        - input: tensor to reverse
        - dim: dimension to reverse on
    Returns:
        - reversed input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return input.index_select(dim, reverse_index)


def rcumsum(input, dim=0):
    """
    Computes a reverse cumulative sum
    Args:
        - input: tensor
        - dim: dimension to reverse on
    Returns:
        - rcumsum on input
    """
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return torch.index_select(input, dim, reverse_index).cumsum(dim).index_select(dim, reverse_index)


def approx_equal(self, other, epsilon=1e-4):
    """
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
    """
    if isinstance(self, Variable):
        self = self.data
    if isinstance(other, Variable):
        other = other.data
    return torch.max((self - other).abs()) <= epsilon


__all__ = [LBFGS, pd_catcher]
