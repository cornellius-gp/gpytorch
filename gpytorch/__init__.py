import torch
from torch.autograd import Variable
from .lazy import LazyVariable, ToeplitzLazyVariable
from .module import Module
from .gp_model import GPModel
from .functions import AddDiag, DSMM, NormalCDF, LogNormalCDF
from .utils import function_factory
from .posterior import DefaultPosteriorStrategy


_invmm_class = function_factory.invmm_factory()
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
        return mat.add(1e-3 * Variable(torch.eye(len(mat))))
    else:
        return mat.add(1e-3 * torch.eye(len(mat)))


def dsmm(sparse_mat, dense_mat):
    return DSMM(sparse_mat)(dense_mat)


def exact_gp_marginal_log_likelihood(covar, target, num_samples=10):
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
        return covar.exact_gp_marginal_log_likelihood(target, num_samples)
    else:
        return _exact_gp_mll_class()(covar, target)


def invmm(mat1, mat2):
    """
    Computes a linear solve with several right hand sides.

    Args:
        - mat1 (matrix nxn) - Matrix to solve with
        - mat2 (matrix nxk) - Matrix of k right hand side vectors.

    Returns:
        - matrix nxk - (mat1)^{-1}mat2
    """
    if isinstance(mat1, LazyVariable):
        return mat1.invmm(mat2)
    else:
        return _invmm_class()(mat1, mat2)


def invmv(mat, vec):
    """
    Computes a linear solve with a single right hand side

    Args:
        - mat1 (matrix nxn) - Matrix to solve with
        - vec (vector n) - Right hand side vector

    Returns:
        - vector n - (mat1)^{-1}vec
    """
    res = invmm(mat, vec.view(-1, 1))
    return res.view(-1)


def log_normal_cdf(x):
    """
    Computes the element-wise log standard normal CDF of an input tensor x.

    This function should always be preferred over calling normal_cdf and taking the log
    manually, as it is more numerically stable.
    """
    return LogNormalCDF()(x)


def monte_carlo_log_likelihood(log_probability_func, train_y,
                               variational_mean, chol_var_covar,
                               train_covar, num_samples):
    if isinstance(train_covar, LazyVariable):
        log_likelihood = train_covar.monte_carlo_log_likelihood(log_probability_func,
                                                                train_y,
                                                                variational_mean,
                                                                chol_var_covar,
                                                                num_samples)
    else:
        epsilon = Variable(torch.randn(len(train_covar), num_samples))
        samples = chol_var_covar.t().mm(epsilon)
        samples = samples + variational_mean.unsqueeze(1)
        log_likelihood = log_probability_func(samples, train_y)

    return log_likelihood


def mvn_kl_divergence(mean_1, chol_covar_1, mean_2, covar_2, num_samples=10):
    """
    PyTorch function for computing the KL-Divergence between two multivariate
    Normal distributions.

    For this function, the first Gaussian distribution is parameterized by the
    mean vector \mu_1 and the Cholesky decomposition of the covariance matrix U_1:
    N(\mu_1, U_1^{\top}U_1).

    The second Gaussian distribution is parameterized by the mean vector \mu_2
    and the full covariance matrix \Sigma_2: N(\mu_2, \Sigma_2)

    The KL divergence between two multivariate Gaussians is given by:

        KL(N_1||N_2) = 0.5 * (Tr(\Sigma_2^{-1}\Sigma_{1}) + (\mu_2 -
            \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) -
            logdet(\Sigma_{1}) - D)

    Where D is the dimensionality of the distributions.
    """
    mu_diffs = mean_2 - mean_1

    if isinstance(covar_2, LazyVariable):
        trace_logdet_quadform = covar_2.trace_log_det_quad_form(mu_diffs, chol_covar_1, num_samples)
    else:
        trace_logdet_quadform = _trace_logdet_quad_form_factory_class(num_samples)(mu_diffs,
                                                                                   chol_covar_1,
                                                                                   covar_2)

    log_det_covar1 = chol_covar_1.diag().log().sum(0) * 2

    # get D
    D = len(mu_diffs)

    # Compute the KL Divergence.
    res = 0.5 * (trace_logdet_quadform - log_det_covar1 - D)

    return res


def normal_cdf(x):
    """
    Computes the element-wise standard normal CDF of an input tensor x.
    """
    return NormalCDF()(x)


def posterior_strategy(obj):
    if isinstance(obj, LazyVariable):
        return obj.posterior_strategy()
    return DefaultPosteriorStrategy(obj)


__all__ = [
    ToeplitzLazyVariable,
    Module,
    GPModel,
    add_diag,
    add_jitter,
    dsmm,
    exact_gp_marginal_log_likelihood,
    invmm,
    invmv,
    log_normal_cdf,
    monte_carlo_log_likelihood,
    mvn_kl_divergence,
    normal_cdf,
    posterior_strategy,
]
