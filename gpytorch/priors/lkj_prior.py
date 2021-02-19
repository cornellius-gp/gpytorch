#!/usr/bin/env python3

import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.nn import Module as TModule

from .. import settings
from ..utils.cholesky import psd_safe_cholesky
from .prior import Prior


class LKJPrior(Prior):
    r"""LKJ prior over n x n (positive definite) correlation matrices

    .. math:

        \begin{equation*}
            pdf(\Sigma) ~ |\Sigma| ^ (\eta  - 1)
        \end{equation*}

    where :math:`\eta > 0` is a shape parameter.

    Reference: Bayesian Data Analysis, 3rd ed., Gelman et al., p. 576
    """

    arg_constraints = {"n": constraints.positive_integer, "eta": constraints.positive}
    # TODO: move correlation matrix validation upstream into pytorch
    support = constraints.positive_definite
    _validate_args = True

    def __init__(self, n, eta, validate_args=False):
        TModule.__init__(self)
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if isinstance(eta, Number):
            eta = torch.tensor(float(eta))
        self.n = torch.tensor(n, dtype=torch.long, device=eta.device)
        batch_shape = eta.shape
        event_shape = torch.Size([n, n])
        # Normalization constant(s)
        i = torch.arange(n, dtype=eta.dtype, device=eta.device)
        C = (((2 * eta.view(-1, 1) - 2 + i) * i).sum(1) * math.log(2)).view_as(eta)
        C += n * torch.sum(2 * torch.lgamma(i / 2 + 1) - torch.lgamma(i + 2))
        # need to assign values before registering as buffers to make argument validation work
        self.eta = eta
        self.C = C
        super(LKJPrior, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.eta, self.C
        self.register_buffer("eta", eta)
        self.register_buffer("C", C)

    def log_prob(self, X):
        if any(s != self.n for s in X.shape[-2:]):
            raise ValueError("Correlation matrix is not of size n={}".format(self.n.item()))
        if not _is_valid_correlation_matrix(X):
            raise ValueError("Input is not a valid correlation matrix")
        log_diag_sum = psd_safe_cholesky(X, upper=True).diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return self.C + (self.eta - 1) * 2 * log_diag_sum


class LKJCholeskyFactorPrior(LKJPrior):
    r"""LKJ prior over n x n (positive definite) Cholesky-decomposed
    correlation matrices

    .. math:

        \begin{equation*}
            pdf(\Sigma) ~ |\Sigma| ^ (\eta  - 1)
        \end{equation*}

    where :math:`\eta > 0` is a shape parameter and n is the dimension of the
    correlation matrix.

    LKJCholeskyFactorPrior is different from LKJPrior in that it accepts the
    Cholesky factor of the correlation matrix to compute probabilities.
    """

    support = constraints.lower_cholesky

    def log_prob(self, X):
        if any(s != self.n for s in X.shape[-2:]):
            raise ValueError("Cholesky factor is not of size n={}".format(self.n.item()))
        if not _is_valid_correlation_matrix_cholesky_factor(X):
            raise ValueError("Input is not a Cholesky factor of a valid correlation matrix")
        log_diag_sum = torch.diagonal(X, dim1=-2, dim2=-1).log().sum(-1)
        return self.C + (self.eta - 1) * 2 * log_diag_sum


class LKJCovariancePrior(LKJPrior):
    """LKJCovariancePrior combines an LKJ prior over the correlation matrix
    and a user-specified prior over marginal standard deviations to return a
    prior over the full covariance matrix.

    Usage: LKJCovariancePrior(n, eta, sd_prior), where
        n is a positive integer, the size of the covariance matrix,
        eta is a positive shape parameter for the LKJPrior over correlations, and
        sd_prior is a scalar Prior over nonnegative numbers, which is used for
        each of the n marginal standard deviations on the covariance matrix.
    """

    def __init__(self, n, eta, sd_prior, validate_args=False):
        if not isinstance(sd_prior, Prior):
            raise ValueError("sd_prior must be an instance of Prior")
        if not isinstance(n, int):
            raise ValueError("n must be an integer")
        if sd_prior.event_shape not in {torch.Size([1]), torch.Size([n])}:
            raise ValueError("sd_prior must have event_shape 1 or n")
        correlation_prior = LKJPrior(n=n, eta=eta, validate_args=validate_args)
        if sd_prior.batch_shape != correlation_prior.batch_shape:
            raise ValueError("sd_prior must have same batch_shape as eta")
        TModule.__init__(self)
        super(LKJPrior, self).__init__(
            correlation_prior.batch_shape, correlation_prior.event_shape, validate_args=False
        )
        self.correlation_prior = correlation_prior
        self.sd_prior = sd_prior

    def log_prob(self, X):
        marginal_var = torch.diagonal(X, dim1=-2, dim2=-1)
        if not torch.all(marginal_var >= 0):
            raise ValueError("Variance(s) cannot be negative")
        marginal_sd = marginal_var.sqrt()
        sd_diag_mat = _batch_form_diag(1 / marginal_sd)
        correlations = torch.matmul(torch.matmul(sd_diag_mat, X), sd_diag_mat)
        log_prob_corr = self.correlation_prior.log_prob(correlations)
        log_prob_sd = self.sd_prior.log_prob(marginal_sd)
        return log_prob_corr + log_prob_sd


def _batch_form_diag(tsr):
    """Form diagonal matrices in batch mode."""
    eye = torch.eye(tsr.shape[-1], dtype=tsr.dtype, device=tsr.device)
    M = tsr.unsqueeze(-1).expand(tsr.shape + tsr.shape[-1:])
    return eye * M


def _is_valid_correlation_matrix(Sigma, tol=1e-6):
    """Check if supplied matrix is a valid correlation matrix

    A matrix is a valid correlation matrix if it is positive semidefinite, and
    if all diagonal elements are equal to 1.

    Args:
        Sigma: A n x n correlation matrix, or a batch of b correlation matrices
            with shape b x n x n
        tol: The tolerance with which to check unit value of the diagonal elements

    Returns:
        True if Sigma is a valid correlation matrix, False otherwise (in batch
            mode, all matrices in the batch need to be valid correlation matrices)

    """
    if settings.verbose_linalg.on():
        settings.verbose_linalg.logger.debug(f"Running symeig on a matrix of size {Sigma.shape}.")

    evals, _ = torch.symeig(Sigma, eigenvectors=False)
    if not torch.all(evals >= -tol):
        return False
    return all(torch.all(torch.abs(S.diag() - 1) < tol) for S in Sigma.view(-1, *Sigma.shape[-2:]))


def _is_valid_correlation_matrix_cholesky_factor(L, tol=1e-6):
    """Check if supplied matrix is a Cholesky factor of a valid correlation matrix

    A matrix is a Cholesky fator of a valid correlation matrix if it is lower
    triangular, has positive diagonal, and unit row-sum

    Args:
        L: A n x n lower-triangular matrix, or a batch of b lower-triangular
            matrices with shape b x n x n
        tol: The tolerance with which to check positivity of the diagonal and
            unit-sum of the rows

    Returns:
        True if L is a Cholesky factor of a valid correlation matrix, False
            otherwise (in batch mode, all matrices in the batch need to be
            Cholesky factors of valid correlation matrices)

    """
    unit_row_length = torch.all((torch.norm(L, dim=-1) - 1).abs() < tol)
    return unit_row_length and torch.all(constraints.lower_cholesky.check(L))
