"""Cholesky-based Gaussian process approximation strategy."""

from __future__ import annotations

from jaxtyping import Float
from linear_operator import operators
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
    ) -> None:
        super().init_cache(model=model)

        # Buffers which cache repeatedly used quantities
        self.register_cached_quantity(
            "prior_predictive_train_mean",
            quantity=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )
        self.register_cached_quantity(
            "prior_predictive_train_covariance_cholesky_decomposition",
            quantity=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )
        self.register_cached_quantity(
            "representer_weights",
            quantity=None,
            persistent=True,
            clear_cache_on=["backward", "set_train_inputs"],
        )

    def _cache_prior_predictive_mean_and_covariance_cholesky_decomposition(
        self,
    ) -> None:
        """Computes prior predictive mean and Cholesky factor and fills the respective buffers."""
        prior_predictive_train = self.model.likelihood(self.model.forward(self.model.train_inputs))
        self.prior_predictive_train_mean = prior_predictive_train.mean
        self.prior_predictive_train_covariance_cholesky_decomposition = operators.CholLinearOperator(
            prior_predictive_train.lazy_covariance_matrix.cholesky()
        )

    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # Prior at test inputs
        prior_mean_test = self.model.mean_module(inputs)
        prior_covariance_test = self.model.covar_module(inputs)

        # Prior predictive at training inputs
        if self.prior_predictive_train_covariance_cholesky_decomposition is None:
            self._cache_prior_predictive_mean_and_covariance_cholesky_decomposition()

        # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
        covariance_train_test = self.model.covar_module(self.model.train_inputs, inputs).to_dense()
        covariance_test_train_inverse_cholesky_factor = (
            self.prior_predictive_train_covariance_cholesky_decomposition.cholesky()
            .solve(covariance_train_test)
            .transpose(-2, -1)
        )

        # Posterior mean evaluated at test inputs
        if self.representer_weights is None:
            self.representer_weights = self.prior_predictive_train_covariance_cholesky_decomposition.solve(
                (self.model.train_targets - self.prior_predictive_train_mean).unsqueeze(-1)
            ).squeeze(-1)

        posterior_mean_test = prior_mean_test + covariance_train_test.transpose(-2, -1) @ self.representer_weights

        # Posterior covariance evaluated at test inputs
        posterior_covariance_test = prior_covariance_test - operators.RootLinearOperator(
            covariance_test_train_inverse_cholesky_factor
        )

        return distributions.MultivariateNormal(posterior_mean_test, posterior_covariance_test)
