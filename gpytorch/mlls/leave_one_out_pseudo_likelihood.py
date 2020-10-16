#!/usr/bin/env python3
import math

import torch
from torch import Tensor

from ..distributions import MultivariateNormal
from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood


class LeaveOneOutPseudoLikelihood(ExactMarginalLogLikelihood):
    """
    The leave one out cross-validation (LOO-CV) likelihood from RW 5.4.2 for an exact Gaussian process with a
    Gaussian likelihood. This offers an alternative to the exact marginal log likelihood where we
    instead maximize the sum of the leave one out log probabilities :math:`\log p(y_i | X, y_{-i}, \theta)`.

    Naively, this will be O(n^4) with Cholesky as we need to compute `n` Cholesky factorizations. Fortunately,
    given the Cholesky factorization of the full kernel matrix (without any points removed), we can compute
    both the mean and variance of each removed point via a bordered system formulation making the total
    complexity O(n^3).

    The LOO-CV approach can be more robust against model mis-specification as it gives an estimate for the
    (log) predictive probability, whether or not the assumptions of the model is fulfilled.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -loocv(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        super().__init__(likelihood=likelihood, model=model)
        self.likelihood = likelihood
        self.model = model

    def forward(self, function_dist: MultivariateNormal, target: Tensor, *params) -> Tensor:
        r"""
        Computes the leave one out likelihood given :math:`p(\mathbf f)` and `\mathbf y`

        :param ~gpytorch.distributions.MultivariateNormal output: the outputs of the latent function
            (the :obj:`~gpytorch.models.GP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param dict kwargs: Additional arguments to pass to the likelihood's :attr:`forward` function.
        """
        output = self.likelihood(function_dist, *params)
        m, K = output.mean, output.covariance_matrix
        m = m.reshape(*target.shape)
        L = torch.cholesky(K, upper=False)
        I = torch.eye(*K.shape[-2:], dtype=K.dtype, device=K.device)
        sigma2 = 1.0 / torch.cholesky_solve(I, L, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
        mu = target - torch.cholesky_solve((target - m).unsqueeze(-1), L, upper=False).squeeze(-1) * sigma2
        term1 = -0.5 * torch.log(sigma2)
        term2 = -0.5 * (target - mu).pow(2.0) / sigma2
        log_loocv = term1 + term2 - 0.5 * math.log(2 * math.pi)
        res = log_loocv.sum(dim=-1)

        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        for _, prior, closure, _ in self.named_priors():
            res.add_(prior.log_prob(closure()).sum())

        # Scale by the amount of data we have
        num_data = target.size(-1)
        return res.div_(num_data)
