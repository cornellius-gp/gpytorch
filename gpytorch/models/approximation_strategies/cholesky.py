"""Cholesky-based Gaussian process approximation strategy."""

from __future__ import annotations

from typing import Optional

import torch
from jaxtyping import Float
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from ... import distributions, Module
from .approximation_strategy import ApproximationStrategy


class Cholesky(ApproximationStrategy):
    """Approximation strategy using a Cholesky decomposition."""

    def __init__(self) -> None:
        super().__init__()

    def init_cache(
        self,
        model: Module,
        train_inputs: Optional[Float[Tensor, "N D"]] = None,
        train_targets: Optional[Float[Tensor, " N"]] = None,
    ) -> None:
        super().init_cache(model, train_inputs, train_targets)

        # Buffers which cache repeatedly used quantities
        self.register_cached_quantity(
            "prior_predictive_train_mean",
            tensor=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )
        self.register_cached_quantity(
            "prior_predictive_train_covariance_cholesky_factor",
            tensor=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )
        self.register_cached_quantity(
            "representer_weights",
            tensor=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )

    def _cache_prior_predictive_mean_and_covariance_cholesky_factor(
        self,
    ) -> None:
        """Computes prior predictive mean and Cholesky factor and fills the respective buffers."""
        prior_predictive_train = self.model.likelihood(self.model.forward(self.train_inputs))
        self.prior_predictive_train_mean = prior_predictive_train.mean
        # self.prior_predictive_train_covariance_cholesky_factor = psd_safe_cholesky(
        #     prior_predictive_train.covariance_matrix
        # )
        self.prior_predictive_train_covariance_cholesky_factor = (
            prior_predictive_train.lazy_covariance_matrix.cholesky()
        )
        # TODO: How can we cache linear operators in buffers?

    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Prior at test inputs
        prior_mean_test = self.model.mean_module(inputs)
        prior_covariance_test = self.model.covar_module(inputs)

        # Prior predictive at training inputs
        if self.prior_predictive_train_covariance_cholesky_factor is None:
            self._cache_prior_predictive_mean_and_covariance_cholesky_factor()

        # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
        covariance_train_test = self.model.covar_module(self.train_inputs, inputs).to_dense()
        covariance_test_train_inverse_cholesky_factor = torch.linalg.solve_triangular(
            self.prior_predictive_train_covariance_cholesky_factor, covariance_train_test, upper=False
        ).transpose(-2, -1)

        # Posterior mean evaluated at test inputs
        if self.representer_weights is None:
            self.representer_weights = torch.cholesky_solve(
                (self.train_targets - self.prior_predictive_train_mean).unsqueeze(-1),
                self.prior_predictive_train_covariance_cholesky_factor,
                upper=False,
            ).squeeze(-1)
        posterior_mean_test = prior_mean_test + covariance_train_test.transpose(-2, -1) @ self.representer_weights

        # Posterior covariance evaluated at test inputs
        posterior_covariance_test = prior_covariance_test - operators.RootLinearOperator(
            covariance_test_train_inverse_cholesky_factor
        )

        return distributions.MultivariateNormal(posterior_mean_test, posterior_covariance_test)
