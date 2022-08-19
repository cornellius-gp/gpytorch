#!/usr/bin/env python3

import torch
from torch.distributions import LKJCholesky, constraints
from torch.nn import Module as TModule

from .. import settings
from ..utils.cholesky import psd_safe_cholesky
from .prior import Prior


class LKJCholeskyFactorPrior(Prior, LKJCholesky):
    r"""LKJ prior over n x n (positive definite) Cholesky-decomposed
    correlation matrices

    .. math:

        \begin{equation*}
            pdf(\Sigma) \sim |\Sigma| ^ (\eta  - 1)
        \end{equation*}

    where :math:`\eta > 0` is a shape parameter and n is the dimension of the
    correlation matrix.

    LKJCholeskyFactorPrior is different from LKJPrior in that it accepts the
    Cholesky factor of the correlation matrix to compute probabilities.

    This distribution assumes that the cholesky factor is lower-triangular.
    """

    def __init__(self, n, eta, validate_args=False, transform=None):
        TModule.__init__(self)
        LKJCholesky.__init__(self, dim=n, concentration=eta, validate_args=validate_args)
        self.n = self.dim
        self.eta = self.concentration
        self._transform = transform

    # TODO: change from sample to rsample if pytorch #69281 goes in providing rsample method


class LKJPrior(LKJCholeskyFactorPrior):
    r"""LKJ prior over n x n (positive definite) correlation matrices

    .. math:

        \begin{equation*}
            pdf(\Sigma) \sim |\Sigma| ^ (\eta  - 1)
        \end{equation*}

    where :math:`\eta > 0` is a shape parameter.

    Reference: Bayesian Data Analysis, 3rd ed., Gelman et al., p. 576
    """
    support = constraints.positive_definite

    def log_prob(self, X):
        if any(s != self.n for s in X.shape[-2:]):
            raise ValueError("Correlation matrix is not of size n={}".format(self.n))
        if not _is_valid_correlation_matrix(X):
            raise ValueError("Input is not a valid correlation matrix")
        X_cholesky = psd_safe_cholesky(X, upper=False)
        return super().log_prob(X_cholesky)

    def sample(self, sample_shape=torch.Size()):
        R = super().sample(sample_shape=sample_shape)
        return R.matmul(R.transpose(-1, -2))


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
        # bug-fix event shapes if necessary
        if sd_prior.event_shape not in {torch.Size([1]), torch.Size([n]), torch.Size([])}:
            if sd_prior.event_shape == torch.Size([]):
                sd_prior._event_shape = torch.Size([1])
            raise ValueError("sd_prior must have event_shape 1 or n")
        correlation_prior = LKJPrior(n=n, eta=eta, validate_args=validate_args)
        if sd_prior.batch_shape != correlation_prior.batch_shape:
            raise ValueError("sd_prior must have same batch_shape as eta")
        TModule.__init__(self)
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

    def sample(self, sample_shape=torch.Size()):
        base_correlation = self.correlation_prior.sample(sample_shape)
        marginal_sds = self.sd_prior.rsample(sample_shape)
        # expand sds to have the same shape as the base correlation matrix
        marginal_sds = marginal_sds.repeat(*[1] * len(sample_shape), self.correlation_prior.n)
        marginal_sds = torch.diag_embed(marginal_sds)
        return marginal_sds.matmul(base_correlation).matmul(marginal_sds)


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

    evals = torch.linalg.eigvalsh(Sigma)
    if not torch.all(evals >= -tol):
        return False
    return all(torch.all(torch.abs(S.diagonal(dim1=-1, dim2=-2) - 1) < tol) for S in Sigma.view(-1, *Sigma.shape[-2:]))


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
