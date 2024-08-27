"""Cholesky-based Gaussian process approximation strategy."""

from __future__ import annotations

import torch

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

    def posterior(self, inputs: Float[Tensor, "M D"]) -> distributions.MultivariateNormal:

        # TODO: leverage this code:
        # # Concatenate the input to the training input
        # train_batch_shape = self.model.train_inputs.shape[:-2]
        # # Make sure the batch shapes agree for training/test data
        # if train_batch_shape != self.model.train_inputs.shape[:-2]:
        #     train_batch_shape = torch.broadcast_shapes(train_batch_shape, self.model.train_inputs.shape[:-2])
        #     self.model.train_inputs = self.model.train_inputs.expand(
        #         *train_batch_shape, *self.model.train_inputs.shape[-2:]
        #     )
        # if train_batch_shape != input.shape[:-2]:
        #     train_batch_shape = torch.broadcast_shapes(train_batch_shape, input.shape[:-2])
        #     self.model.train_inputs = self.model.train_inputs.expand(
        #         *train_batch_shape, *self.model.train_inputs.shape[-2:]
        #     )
        #     input = input.expand(*train_batch_shape, *input.shape[-2:])
        # all_inputs = torch.cat([self.model.train_inputs, input], dim=-2)

        # Prior at train and test inputs
        prior_train_test = self.model.prior(torch.cat((self.model.train_inputs, inputs), dim=-2))
        prior_train = prior_train_test[..., 0 : self.model.train_inputs.shape[-2]]
        prior_test = prior_train_test[..., -inputs.shape[-2] :]
        covariance_train_test = prior_train_test.lazy_covariance_matrix[
            ..., 0 : self.model.train_inputs.shape[-2], -inputs.shape[-2] :
        ].to_dense()

        # Prior predictive and Cholesky decomposition of Gramian
        if self.prior_predictive_train_covariance_cholesky_decomposition is None:
            prior_predictive_train = self.model.likelihood(prior_train)

            self.prior_predictive_train_covariance_cholesky_decomposition = operators.CholLinearOperator(
                prior_predictive_train.lazy_covariance_matrix.cholesky()
            )

        # Matrix-square root of the covariance downdate: k(x, X)L^{-1}
        covariance_test_train_inverse_cholesky_factor = (
            self.prior_predictive_train_covariance_cholesky_decomposition.cholesky()
            .solve(covariance_train_test)
            .transpose(-2, -1)
        )

        # Posterior mean evaluated at test inputs
        if self.representer_weights is None:
            self.representer_weights = self.prior_predictive_train_covariance_cholesky_decomposition.solve(
                (self.model.train_targets - prior_train.mean).unsqueeze(-1)
            ).squeeze(-1)

        posterior_mean_test = prior_test.mean + (
            covariance_train_test.transpose(-2, -1) @ self.representer_weights.unsqueeze(-1)
        ).squeeze(-1)

        # Posterior covariance evaluated at test inputs
        posterior_covariance_test = prior_test.lazy_covariance_matrix - operators.RootLinearOperator(
            covariance_test_train_inverse_cholesky_factor
        )

        return distributions.MultivariateNormal(posterior_mean_test, posterior_covariance_test)
