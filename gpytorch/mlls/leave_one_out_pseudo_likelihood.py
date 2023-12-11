#!/usr/bin/env python3
import math

import torch
from torch import Tensor

from ..distributions import MultivariateNormal
from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood, MarginalLogLikelihood


class LeaveOneOutPseudoLikelihood(ExactMarginalLogLikelihood):
    r"""
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
        :param dict kwargs: Additional arguments to pass to the likelihood's forward function.
        """
        output = self.likelihood(function_dist, *params)
        m, L = output.mean, output.lazy_covariance_matrix.cholesky(upper=False)
        m = m.reshape(*target.shape)
        identity = torch.eye(*L.shape[-2:], dtype=m.dtype, device=m.device)
        sigma2 = 1.0 / L._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2)  # 1 / diag(inv(K))
        mu = target - L._cholesky_solve((target - m).unsqueeze(-1), upper=False).squeeze(-1) * sigma2
        term1 = -0.5 * sigma2.log()
        term2 = -0.5 * (target - mu).pow(2.0) / sigma2
        res = (term1 + term2).sum(dim=-1)

        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have and then add on the scaled constant
        num_data = target.size(-1)
        return res.div_(num_data) - 0.5 * math.log(2 * math.pi)


class LeaveOneActionOutPseudoLikelihood(MarginalLogLikelihood):
    r"""
    The leave-one-(action)-out cross-validation (LOO-CV) likelihood of a computation-aware Gaussian process
    with a Gaussian likelihood.

    This offers an alternative to the exact marginal log likelihood where we
    instead maximize the sum of the leave one *action* out log probabilities
    :math:`\log p(s_j^{\\top} y | X, S_{-j}^{\\top} y, \\theta)`.

    The LOO-CV approach can be more robust against model misspecification as it gives an estimate for the
    (log) predictive probability, whether or not the assumptions of the model is fulfilled.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ComputationAwareGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model.
    :param ~gpytorch.models.ComputationAwareGP model: The computation-aware iterative GP model.

    Example:
        >>> # model is a gpytorch.models.ComputationAwareGP
        >>> # likelihood is a gpytorch.likelihoods.GaussianLikelihood
        >>> loocv = gpytorch.mlls.ComputationAwareLeaveOneOutPseudoLikelihood(likelihood, model)
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
        Computes the leave-one-action-out likelihood given :math:`p(\mathbf f)` and `\mathbf y`

        :param ~gpytorch.distributions.MultivariateNormal function_dist: the distribution of the latent function
            (the :obj:`~gpytorch.models.GP`) evaluated at the data.
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :param dict kwargs: Additional arguments to pass to the likelihood's forward function.
        """
        Khat = self.likelihood(function_dist).lazy_covariance_matrix.evaluate_kernel()

        with torch.no_grad():
            solver_state = self.model.linear_solver.solve(Khat, target)
            if self.model.prediction_strategy is None:
                self.model._solver_state = solver_state

        actions = solver_state.cache["actions"]

        # Naive implementation
        loocv_objective = torch.zeros(())
        jitter = 1e-6  # TODO: do we really need this?

        Khat_actions = Khat @ actions
        actions_Khat_actions = Khat_actions.T @ actions
        actions_K_actions = actions_Khat_actions - self.likelihood.noise.item() * actions.T @ actions
        prior_mean = torch.zeros_like(target)  # TODO: assumes zero prior mean
        actions_targets_minus_mean = actions.T @ (target - prior_mean)
        num_actions = actions.shape[1]

        for j in range(num_actions):
            idcs_loo = torch.ones(num_actions).type(torch.bool)
            idcs_loo[j] = False

            # Components of low-rank precision approximation C_{-j}
            actions_loo_Khat_actions_loo = actions_Khat_actions[idcs_loo, :][:, idcs_loo]
            L_loo = torch.linalg.cholesky(
                actions_loo_Khat_actions_loo + jitter * torch.eye(num_actions - 1), upper=False
            )

            # Mean prediction error
            pred_error = actions_targets_minus_mean[j] - actions_K_actions[j, :][idcs_loo] @ torch.cholesky_solve(
                actions_targets_minus_mean[idcs_loo][:, None], L_loo, upper=False
            )

            # Predictive covariance for s_j'y
            pred_cov = actions_Khat_actions[j, :][j] - actions_K_actions[j, :][idcs_loo] @ (
                torch.cholesky_solve(actions_K_actions[idcs_loo, :][:, j][:, None], L_loo, upper=False)
            )

            # Log-predictive density: log p(s_j'y | X, S_{-j}'y, \theta)
            log_pred_loo = -0.5 * (torch.sum(pred_error**2) / pred_cov + torch.log(pred_cov) + math.log(2 * math.pi))

            loocv_objective = loocv_objective + log_pred_loo

        return loocv_objective / num_actions
