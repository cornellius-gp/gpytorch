from .module import Module
from . import models
from . import means
from . import mlls
from . import kernels
from . import beta_features
from . import settings
from .mlls import ExactMarginalLogLikelihood, VariationalMarginalLogLikelihood
from .beta_features import fast_pred_var
from torch.autograd import Variable
from .lazy import LazyVariable, NonLazyVariable
from .functions import add_diag as _add_diag, dsmm, log_normal_cdf, normal_cdf
from .utils import function_factory


_inv_matmul_class = function_factory.inv_matmul_factory()
_inv_quad_log_det_class = function_factory.inv_quad_log_det_factory()


def add_diag(input, diag):
    """
    Adds a diagonal matrix s*I to the input matrix input.

    Args:
        - input (matrix nxn) - Variable or LazyVariable wrapping matrix to add diagonal \
                               component to.
        - diag (scalar) - Scalar s so that s*I is added to the input matrix.

    Returns:
        - matrix nxn - Variable or LazyVariable wrapping a new matrix with the diagonal \
                       component added.
    """
    if not isinstance(diag, Variable):
        raise RuntimeError('Expected a variable for the diagonal component.')

    if isinstance(input, LazyVariable):
        return input.add_diag(diag)
    else:
        return _add_diag(input, diag)


def add_jitter(mat):
    """
    Adds "jitter" to the diagonal of a matrix.
    This ensures that a matrix that *should* be positive definite *is* positive definate.

    Args:
        - mat (matrix nxn) - Positive definite matrxi
    Returns: (matrix nxn)
    """
    if isinstance(mat, LazyVariable):
        return mat.add_jitter()
    elif isinstance(mat, Variable):
        diag = Variable(mat.data.new(mat.size(-1)).fill_(1e-3).diag())
        if mat.ndimension() == 3:
            return mat + diag.unsqueeze(0).expand(mat.size(0), mat.size(1), mat.size(2))
        else:
            return mat + diag
    else:
        diag = mat.new(mat.size(-1)).fill_(1e-3).diag()
        if mat.ndimension() == 3:
            return mat.add_(diag.unsqueeze(0).expand(mat.size(0), mat.size(1), mat.size(2)))
        else:
            return diag.add_(mat)


def exact_predictive_mean(full_covar, full_mean, train_labels, noise, precomputed_cache=None):
    """
    Computes the posterior predictive mean of a GP

    Args:
    - full_covar ( (n+t) x (n+t) ) - the block prior covariance matrix of training and testing points
        - [ K_XX, K_XX*; K_X*X, K_X*X* ]
    - full_mean (n + t) - the training and test prior means, stacked on top of each other
    - train_labels (n) - the training labels minus the training prior mean
    - noise (1) - the observed noise (from the likelihood)
    - precomputed_cache - speeds up subsequent computations (default: None)

    Returns:
    - (t) - the predictive posterior mean of the test points
    """
    if not isinstance(full_covar, LazyVariable):
        full_covar = NonLazyVariable(full_covar)
    return full_covar.exact_predictive_mean(full_mean, train_labels, noise, precomputed_cache)


def exact_predictive_covar(full_covar, n_train, noise, precomputed_cache=None):
    """
    Computes the posterior predictive covariance of a GP

    Args:
    - full_covar ( (n+t) x (n+t) ) - the block prior covariance matrix of training and testing points
        - [ K_XX, K_XX*; K_X*X, K_X*X* ]
    - n_train (int) - how many training points are there in the full covariance matrix
    - noise (1) - the observed noise (from the likelihood)
    - precomputed_cache - speeds up subsequent computations (default: None)

    Returns:
    - LazyVariable (t x t) - the predictive posterior covariance of the test points
    """
    if not isinstance(full_covar, LazyVariable):
        full_covar = NonLazyVariable(full_covar)
    return full_covar.exact_predictive_covar(n_train, noise, precomputed_cache)


def inv_matmul(mat1, rhs):
    """
    Computes a linear solve with several right hand sides.

    Args:
        - mat1 (matrix nxn) - Matrix to solve with
        - rhs (matrix nxk) - rhs matrix or vector

    Returns:
        - matrix nxk - (mat1)^{-1} rhs
    """
    if isinstance(mat1, LazyVariable):
        return mat1.inv_matmul(rhs)
    else:
        return _inv_matmul_class()(mat1, rhs)


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
    if isinstance(mat, LazyVariable):
        return mat.inv_quad_log_det(inv_quad_rhs=inv_quad_rhs, log_det=log_det)
    else:
        batch_size = mat.size(0) if mat.ndimension() == 3 else None
        if inv_quad_rhs is None:
            return _inv_quad_log_det_class(matrix_size=mat.size(-1), batch_size=batch_size,
                                           tensor_cls=mat.new, inv_quad=False,
                                           log_det=log_det)(mat)
        else:
            return _inv_quad_log_det_class(matrix_size=mat.size(-1), batch_size=batch_size,
                                           tensor_cls=mat.new, inv_quad=True,
                                           log_det=log_det)(mat, inv_quad_rhs)


def log_det(mat):
    """
    Computes an (approximate) log determinant of the matrix

    Returns:
        - scalar - log determinant
    """
    _, res = inv_quad_log_det(mat, inv_quad_rhs=None, log_det=True)
    return res


__all__ = [
    # Submodules
    models,
    mlls,
    means,
    kernels,
    # Classes
    Module,
    ExactMarginalLogLikelihood,
    VariationalMarginalLogLikelihood,
    # Functions
    add_diag,
    add_jitter,
    dsmm,
    exact_predictive_mean,
    exact_predictive_covar,
    inv_matmul,
    inv_quad,
    inv_quad_log_det,
    log_det,
    log_normal_cdf,
    normal_cdf,
    # Context managers
    beta_features,
    fast_pred_var,
    settings,
]
