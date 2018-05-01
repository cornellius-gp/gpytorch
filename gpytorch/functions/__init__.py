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


def inv_matmul(mat, rhs):
    """
    Computes a linear solve with several right hand sides.

    Args:
        - mat (matrix nxn) - Matrix to solve with
        - rhs (matrix nxk) - rhs matrix or vector

    Returns:
        - matrix nxk - (mat)^{-1} rhs
    """
    # Does the mat have its own way to do inv_matmuls?
    if hasattr(mat, 'inv_matmul'):
        return mat.inv_matmul(rhs)
    else:
        from ..lazy.non_lazy_variable import NonLazyVariable
        return NonLazyVariable(mat).inv_matmul(rhs)


def inv_quad(mat, tensor):
    """
    Computes an inverse quadratic form (w.r.t mat) with several right hand sides.
    I.e. computes tr( tensor^T mat^{-1} tensor )

    Args:
        - tensor (tensor nxk) - Vector (or matrix) for inverse quad

    Returns:
        - tensor - tr( tensor^T (mat)^{-1} tensor )
    """
    res, _ = inv_quad_log_det(mat, inv_quad_rhs=tensor, log_det=False)
    return res


def inv_quad_log_det(mat, inv_quad_rhs=None, log_det=False):
    """
    Computes an inverse quadratic form (w.r.t mat) with several right hand sides.
    I.e. computes tr( tensor^T mat^{-1} tensor )
    In addition, computes an (approximate) log determinant of the the matrix

    Args:
        - tensor (tensor nxk) - Vector (or matrix) for inverse quad

    Returns:
        - scalar - tr( tensor^T (mat)^{-1} tensor )
        - scalar - log determinant
    """
    # Does the mat have its own way to do inv_matmuls?
    if hasattr(mat, 'inv_quad_log_det'):
        return mat.inv_quad_log_det(inv_quad_rhs, log_det)
    else:
        from ..lazy.non_lazy_variable import NonLazyVariable
        return NonLazyVariable(mat).inv_quad_log_det(inv_quad_rhs, log_det)


def log_det(mat):
    """
    Computes an (approximate) log determinant of the matrix

    Returns:
        - scalar - log determinant
    """
    _, res = inv_quad_log_det(mat, inv_quad_rhs=None, log_det=True)
    return res


def normal_cdf(x):
    """
    Computes the element-wise standard normal CDF of an input tensor x.
    """
    return NormalCDF()(x)


__all__ = [
    add_diag,
    dsmm,
    inv_matmul,
    inv_quad_log_det,
    log_normal_cdf,
    normal_cdf,
]
