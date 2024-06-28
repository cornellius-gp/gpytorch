"""Cholesky-based Gaussian process approximation strategy."""

from __future__ import annotations

import torch
from jaxtyping import Float
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from ... import distributions, utils
from .approximation_strategy import ApproximationStrategy


class Cholesky(ApproximationStrategy):
    def __init__(self) -> None:
        super().__init__()

    @property
    def prior_predictive_train(self) -> distributions.MultivariateNormal:
        return self.likelihood(self.prior(self.train_inputs))

    @property
    @utils.memoize.cached()
    def prior_mean_train(self) -> Float[Tensor, " N"]:
        return self.prior_predictive_train.mean

    @property
    @utils.memoize.cached()
    def predictive_covariance_train(self) -> Float[Tensor, "N N"]:
        return self.prior_predictive_train.lazy_covariance_matrix.to_dense()

    @property
    @utils.memoize.cached()
    def predictive_covariance_train_cholesky_factor(self) -> Float[Tensor, "N N"]:
        return psd_safe_cholesky(self.predictive_covariance_train)

    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Prior at test inputs
        prior_mean_test = self.mean(inputs)
        prior_covariance_test = self.kernel(inputs)

        # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
        covariance_train_test = self.kernel(self.train_inputs, inputs).to_dense()
        covariance_test_train_inverse_cholesky_factor = torch.linalg.solve_triangular(
            self.predictive_covariance_train_cholesky_factor, covariance_train_test, upper=False
        ).transpose(-2, -1)

        # Posterior mean evaluated at test inputs
        train_targets_offset = self.train_targets - self.prior_mean_train
        posterior_mean_test = prior_mean_test + covariance_train_test.transpose(-2, -1) @ torch.cholesky_solve(
            train_targets_offset.unsqueeze(-1), self.predictive_covariance_train_cholesky_factor, upper=False
        ).squeeze(-1)

        # Posterior covariance evaluated at test inputs
        posterior_covariance_test = prior_covariance_test - operators.RootLinearOperator(
            covariance_test_train_inverse_cholesky_factor
        )

        return distributions.MultivariateNormal(posterior_mean_test, posterior_covariance_test)
