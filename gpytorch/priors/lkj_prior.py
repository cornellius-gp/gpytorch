from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
from torch.distributions.constraints import positive_definite
from gpytorch.priors.prior import Prior


class LKJPrior(Prior):
    """LKJ prior over n x n (positive definite) correlation matrices

    pdf(Sigma) ~ |Sigma| ^ (eta  - 1)

    where eta > 0 is a shape parameter and n is the dimension of the
    correlation matrix.
    """

    def __init__(self, n, eta=1):
        if eta <= 0:
            raise ValueError("Shape parameter of LKJ prior must be positive")

        if n < 1:
            raise ValueError("Dimension n must be a positive integer")
        super(LKJPrior, self).__init__()
        self.register_buffer("eta", torch.tensor(eta))
        self.register_buffer("n", torch.tensor(n))
        # Normalization constant
        # Reference: Bayesian Data Analysis, 3rd ed., Gelman et al., p. 576
        i = torch.arange(n).float()
        C = torch.sum((2 * eta - 2 + i) * i) * math.log(2) + n * torch.sum(
            2 * torch.lgamma(i / 2 + 1) - torch.lgamma(i + 2)
        )
        self.register_buffer("C", C)
        self._log_transform = False

    def _log_prob(self, parameter):
        if not is_valid_correlation_matrix(parameter):
            raise ValueError("Input is not a valid correlation matrix")
        return self.C + (self.eta - 1) * parameter.potrf().diag().log().sum() * 2

    def size(self):
        return torch.Size([self.n, self.n])


class LKJCholeskyFactorPrior(Prior):
    """LKJ prior over n x n (positive definite) Cholesky-decomposed correlation
    matrices

    pdf(Sigma) ~ |Sigma| ^ (eta  - 1)

    where eta > 0 is a shape parameter and n is the dimension of the
    correlation matrix.

    LKJCholeskyFactorPrior is different from LKJPrior in that it accepts the
    Cholesky factor of the correlation matrix to compute probabilities.
    """

    def __init__(self, n, eta=1):
        LKJPrior.__init__(self, n=n, eta=eta)

    def _log_prob(self, parameter):
        if not is_valid_correlation_matrix(parameter.matmul(parameter.transpose(0, 1))):
            raise ValueError("Input is not a valid correlation matrix")
        Ldiag = parameter.diag()
        return self.C + (self.eta - 1) * 2 * Ldiag.log().sum()

    def size(self):
        return torch.Size([self.n, self.n])


class LKJCovariancePrior(Prior):
    """LKJCovariancePrior combines an LKJ prior over the correlation matrix
    and a user-specified prior over marginal standard deviations to return a
    prior over the full covariance matrix.

    Usage: LKJCovariancePrior(n, eta, sd_prior), where
    n (int) is a positive integer, the size of the covariance matrix,
    eta is a positive shape parameter for the LKJPrior over correlations, and
    sd_prior is a scalar Prior over nonnegative numbers, which is used for
    each of the n marginal standard deviations on the covariance matrix.
    """

    def __init__(self, n, eta, sd_prior):
        correlation_prior = LKJPrior(n, eta)
        super(LKJCovariancePrior, self).__init__()
        self.correlation_prior = correlation_prior
        self.sd_prior = sd_prior
        self._log_transform = False

    def _log_prob(self, parameter):
        marginal_var = parameter.diag()
        if torch.all(marginal_var >= 0):
            marginal_sd = marginal_var.sqrt()
        else:
            raise ValueError("Variance(s) cannot be negative")
        sd_diag_mat = (1 / marginal_sd).diag()
        correlations = sd_diag_mat.matmul(parameter).matmul(sd_diag_mat)
        # log likelihood of correlation matrix
        log_prob = self.correlation_prior._log_prob(correlations)
        # Add log likelihoods of each of the n marginal standard deviations
        # using the specified sd_prior
        for i in range(self.correlation_prior.n):
            log_prob = log_prob + self.sd_prior.log_prob(marginal_sd[i])
        return log_prob

    def size(self):
        return self.correlation_prior.size()


def is_valid_correlation_matrix(Sigma, tol=1e-6):
    """ This function returns true when all diagonal elements of Sigma are
    strictly 1 (in a float sense) and the matrix is positive definite,
    and false otherwise.
    """

    pdef = positive_definite.check(Sigma)
    return bool(torch.all(torch.abs(Sigma.diag() - 1) < tol)) if pdef else False
