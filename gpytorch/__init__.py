import math
import torch
from torch.autograd import Variable
from .lazy import LazyVariable, ToeplitzLazyVariable
from .random_variables import GaussianRandomVariable
from .module import Module
from .gp_model import GPModel
from .functions import AddDiag, ExactGPMarginalLogLikelihood, NormalCDF, LogNormalCDF
from .utils import LinearCG, function_factory
from .utils.toeplitz import index_coef_to_sparse, interpolated_toeplitz_mul, toeplitz_mv


__all__ = [
    ToeplitzLazyVariable,
    Module,
    GPModel,
    AddDiag,
    ExactGPMarginalLogLikelihood,
    NormalCDF,
    LogNormalCDF,
]


_invmm_class = function_factory.invmm_factory()


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
        return covar.gp_marginal_log_likelihood(target)
    else:
        return ExactGPMarginalLogLikelihood()(covar, target)


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


def normal_cdf(x):
    """
    Computes the element-wise standard normal CDF of an input tensor x.
    """
    return NormalCDF()(x)


def log_normal_cdf(x):
    """
    Computes the element-wise log standard normal CDF of an input tensor x.

    This function should always be preferred over calling normal_cdf and taking the log
    manually, as it is more numerically stable.
    """
    return LogNormalCDF()(x)


def mvn_kl_divergence(mean_1, chol_covar_1, mean_2, covar_2):
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

    # ExactGPMarginalLogLikelihood gives us -0.5 * [\mu_2 -
    # \mu_1)\Sigma_{2}^{-1}(\mu_2 - \mu_1) + logdet(\Sigma_{2}) + const]
    # Multiplying that by -2 gives us two of the terms in the KL divergence
    # (plus an unwanted constant that we can subtract out).
    K_part = exact_gp_marginal_log_likelihood(covar_2, mu_diffs)

    # Get logdet(\Sigma_{1})
    log_det_covar1 = chol_covar_1.diag().log().sum(0) * 2

    # Get Tr(\Sigma_2^{-1}\Sigma_{1})
    trace = invmm(covar_2, chol_covar_1.t().mm(chol_covar_1)).trace()

    # get D
    D = len(mu_diffs)

    # Compute the KL Divergence. We subtract out D * log(2 * pi) to get rid
    # of the extra unwanted constant term that ExactGPMarginalLogLikelihood gives us.
    res = 0.5 * (trace - log_det_covar1 - 2 * K_part - (1 + math.log(2 * math.pi)) * D)

    return res


def _exact_predict(test_mean, test_test_covar, train_y, train_mean,
                   train_train_covar, train_test_covar, test_train_covar, alpha=None):
    """
    Computes the predictive mean and variance of a posterior Gaussian process. Additionally computes
    certain posterior parameters that can be cached between rounds of predictions.

        Args:
            - test_mean (vector k) - prior mean values for the test points.
            - test_test_covar (matrix kxk) - prior covariance matrix for the test points.
            - train_y (vector n) - training labels
            - train_mean (vector n) - prior mean values for the training points
            - train_train_covar (matrix nxn) - Variable or LazyVariable representing the train-train covariance matrix.
                                               Usually takes the form (K+sI), where K is the prior covariance matrix and
                                               s is the noise variance.
            - train_test_covar (matrix nxk) - prior covariance matrix between training and test points.
            - test_train_covar (matrix nxk) - prior covariance matrix between test and training points.
                                              Usually, this is simply the transpose of train_test_covar.
            - alpha (vector n) - Coefficients for predictive mean that do not depend on the test points.
                                 If not supplied, this function will compute them.

        Returns:
            - GaussianRandomVariable with the predictive mean and covariance matrix for the test points.
            - alpha (vector n) - Coefficients that can be reused when this function is next called.
    """
    if isinstance(train_train_covar, ToeplitzLazyVariable):
        W_test_left = index_coef_to_sparse(test_train_covar.J_left,
                                           test_train_covar.C_left,
                                           len(test_train_covar.c))
        W_test_right = index_coef_to_sparse(test_train_covar.J_right,
                                            test_train_covar.C_right,
                                            len(test_train_covar.c))
        W_train_left = index_coef_to_sparse(train_train_covar.J_left,
                                            train_train_covar.C_left,
                                            len(train_train_covar.c))
        W_train_right = index_coef_to_sparse(train_train_covar.J_right,
                                             train_train_covar.C_right,
                                             len(train_train_covar.c))
        noise_diag = train_train_covar.added_diag

        def train_mul_closure(v):
            return interpolated_toeplitz_mul(train_train_covar.c.data, v, W_train_left, W_train_right, noise_diag.data)

        def test_train_mul_closure(v):
            return interpolated_toeplitz_mul(test_train_covar.c.data, v, W_test_left, W_test_right)

        # Update test mean
        if alpha is None:
            alpha = LinearCG().solve(train_mul_closure, train_y - train_mean.data).unsqueeze(1)
            alpha = torch.dsmm(W_train_right.t(), alpha).squeeze()
            alpha = toeplitz_mv(train_train_covar.c.data, train_train_covar.c.data, alpha).unsqueeze(1)

        test_mean = Variable(test_mean.data.add(torch.dsmm(W_test_left, alpha).squeeze()))

        # TODO: Add a diagonal only mode / use implicit math
        train_test_covar = train_test_covar.evaluate()
        test_test_covar = test_test_covar.evaluate()

        test_test_covar_correction = test_train_mul_closure(LinearCG().solve(train_mul_closure, train_test_covar))
        test_test_covar = Variable(test_test_covar.sub(test_test_covar_correction))
    else:
        train_y_var = Variable(train_y)
        if alpha is None:
            alpha = invmv(train_train_covar, train_y_var - train_mean)

        test_mean = test_mean.add(torch.mv(test_train_covar, alpha))

        # Update test-test covar
        test_test_covar_correction = torch.mm(test_train_covar, invmm(train_train_covar, train_test_covar))
        test_test_covar = test_test_covar.sub(test_test_covar_correction)

    return GaussianRandomVariable(test_mean, test_test_covar), alpha
