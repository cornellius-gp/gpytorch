from .add_diag import AddDiag
from .dsmm import DSMM
from .normal_cdf import NormalCDF
from .log_normal_cdf import LogNormalCDF


def add_diag(input, diag):
    """
    Adds a diagonal matrix s*I to the input matrix input.

    Args:
        - input (matrix nxn) - Variable wrapping matrix to add diagonal \
                               component to.
        - diag (scalar) - Scalar s so that s*I is added to the input matrix.

    Returns:
        - matrix nxn - Variable wrapping a new matrix with the diagonal \
                       component added.
    """
    return AddDiag()(input, diag)


def dsmm(sparse_mat, dense_mat):
    """
    Performs the (batch) matrix multiplication S x D
    where S is a sparse matrix and D is a dense matrix

    Args:
        - sparse_mat (matrix (b x)mxn) - Variable wrapping sparse matrix
        - dense_mat (matrix (b x)nxo) - Variable wrapping dense matrix

    Returns:
        - matrix (b x)mxo - Result
    """
    return DSMM(sparse_mat)(dense_mat)


def log_normal_cdf(x):
    """
    Computes the element-wise log standard normal CDF of an input tensor x.

    This function should always be preferred over calling normal_cdf and taking the log
    manually, as it is more numerically stable.
    """
    return LogNormalCDF()(x)


def normal_cdf(x):
    """
    Computes the element-wise standard normal CDF of an input tensor x.
    """
    return NormalCDF()(x)


__all__ = [
    add_diag,
    dsmm,
    log_normal_cdf,
    normal_cdf,
]
