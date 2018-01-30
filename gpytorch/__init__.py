from .module import Module
from . import models
from . import means
from . import kernels
from . import beta_features
from .beta_features import fast_pred_var
from torch.autograd import Variable
from .lazy import LazyVariable
from .functions import AddDiag, DSMM, NormalCDF, LogNormalCDF
from .utils import function_factory


_inv_matmul_class = function_factory.inv_matmul_factory()
_trace_logdet_quad_form_factory_class = function_factory.trace_logdet_quad_form_factory()
_exact_gp_mll_class = function_factory.exact_gp_mll_factory()


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
        return AddDiag()(input, diag)


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


def dsmm(sparse_mat, dense_mat):
    return DSMM(sparse_mat)(dense_mat)


def exact_gp_marginal_log_likelihood(covar, target):
    """
    Computes the log marginal likelihood of the data with a GP prior and Gaussian noise model
    given a label vector and covariance matrix.

    Args:
        - covar (matrix nxn) - Variable or LazyVariable representing the covariance matrix of the observations.
                               Usually, this is K + s*I, where s is the noise variance, and K is the prior covariance.
        - target (vector n) - Training label vector.

    Returns:
        - scalar - The marginal log likelihood of the data.
    """
    if isinstance(covar, LazyVariable):
        return covar.exact_gp_marginal_log_likelihood(target)
    else:
        return _exact_gp_mll_class()(covar, target)


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


def trace_logdet_quad_form(mean_diffs, chol_covar_1, covar_2):
    if isinstance(covar_2, LazyVariable):
        return covar_2.trace_log_det_quad_form(mean_diffs, chol_covar_1)
    else:
        return _trace_logdet_quad_form_factory_class()(mean_diffs, chol_covar_1, covar_2)


__all__ = [
    # Submodules
    models,
    means,
    kernels,
    # Classes
    Module,
    # Functions
    add_diag,
    add_jitter,
    dsmm,
    exact_gp_marginal_log_likelihood,
    inv_matmul,
    log_normal_cdf,
    normal_cdf,
    trace_logdet_quad_form,
    # Context managers
    beta_features,
    fast_pred_var,
]
