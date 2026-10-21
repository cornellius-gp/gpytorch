#!/usr/bin/env python3

from __future__ import annotations

import math
import unittest
from collections.abc import Iterator
from itertools import product

import torch
from torch import Tensor

import gpytorch
from gpytorch.variational import (
    BatchDecoupledVariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    LMCVariationalStrategy,
    VariationalStrategy,
)


class _RegularizedStrategy(VariationalStrategy):
    penalty: Tensor

    def kl_divergence(self) -> Tensor:
        return 2 * super().kl_divergence() + self.penalty.square()


class _WrappedModel(gpytorch.models.ApproximateGP):
    def __init__(self, base_class: type[VariationalStrategy], wrapper: str, batch: torch.Size, axis: int) -> None:
        distribution = CholeskyVariationalDistribution(3, batch_shape=batch, mean_init_std=0)
        base = base_class(self, torch.linspace(0.1, 0.9, 3).unsqueeze(-1), distribution, learn_inducing_locations=False)
        if isinstance(base, _RegularizedStrategy):
            base.register_parameter("penalty", torch.nn.Parameter(torch.full(batch, 0.2)))
        if wrapper == "independent":
            strategy = IndependentMultitaskVariationalStrategy(base, num_tasks=batch[axis], task_dim=axis)
        else:
            strategy = LMCVariationalStrategy(base, num_tasks=4, num_latents=batch[axis], latent_dim=axis)
        super().__init__(strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch)
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=batch)

    def forward(self, x: Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class TestWrappedVariationalKL(unittest.TestCase):
    def _cases(self) -> Iterator[tuple[str, torch.Size, int, str, torch.dtype]]:
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        configurations = [
            (wrapper, batch, axis)
            for wrapper in ("independent", "lmc")
            for batch, axis in (((3,), -1), ((2, 3), -1), ((3, 2), -2))
        ]
        for (wrapper, batch, axis), device, dtype in product(configurations, devices, (torch.float32, torch.float64)):
            yield wrapper, torch.Size(batch), axis, device, dtype

    @staticmethod
    def _model(
        base_class: type[VariationalStrategy],
        wrapper: str,
        batch: torch.Size,
        axis: int,
        device: str,
        dtype: torch.dtype,
    ) -> _WrappedModel:
        model = _WrappedModel(base_class, wrapper, batch, axis).to(device=device, dtype=dtype)
        base = model.variational_strategy.base_variational_strategy
        q = base._variational_distribution
        with torch.no_grad():
            q.variational_mean.copy_(torch.tensor([-0.2, 0.1, 0.3], device=device, dtype=dtype))
            factor = torch.tensor([[0.8, 0.0, 0.0], [0.05, 0.9, 0.0], [0.1, 0.03, 1.1]], device=device, dtype=dtype)
            q.chol_variational_covar.copy_(factor)
            base.variational_params_initialized.fill_(1)
        return model

    @staticmethod
    def _gaussian_kl(distribution: CholeskyVariationalDistribution) -> Tensor:
        factor = distribution.chol_variational_covar.tril()
        return 0.5 * (
            distribution.variational_mean.square().sum(-1)
            + factor.square().sum((-1, -2))
            - distribution.num_inducing_points
            - 2 * factor.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
        )

    def test_delegated_custom_kl_value_and_batch_axis(self) -> None:
        for wrapper, batch, axis, device, dtype in self._cases():
            with self.subTest(wrapper=wrapper, batch=batch, axis=axis, device=device, dtype=dtype):
                model = self._model(_RegularizedStrategy, wrapper, batch, axis, device, dtype)
                base = model.variational_strategy.base_variational_strategy
                expected = (2 * self._gaussian_kl(base._variational_distribution) + base.penalty.square()).sum(axis)
                actual = model.variational_strategy.kl_divergence()
                tolerance = 2e-6 if dtype == torch.float32 else 2e-12
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    def test_delegated_custom_kl_parameter_gradients(self) -> None:
        for wrapper, batch, axis, device, dtype in self._cases():
            with self.subTest(wrapper=wrapper, batch=batch, axis=axis, device=device, dtype=dtype):
                model = self._model(_RegularizedStrategy, wrapper, batch, axis, device, dtype)
                base = model.variational_strategy.base_variational_strategy
                leaves = (*base._variational_distribution.parameters(), base.penalty)
                expected = (2 * self._gaussian_kl(base._variational_distribution) + base.penalty.square()).sum(axis)
                actual = model.variational_strategy.kl_divergence()
                gradients = torch.autograd.grad(actual.sum(), leaves, allow_unused=True)
                reference = torch.autograd.grad(expected.sum(), leaves)
                tolerance = 2e-6 if dtype == torch.float32 else 2e-12
                for derivative, target in zip(gradients, reference, strict=True):
                    self.assertIsNotNone(derivative)
                    torch.testing.assert_close(derivative, target, atol=tolerance, rtol=tolerance)

    def test_decoupled_kl_retains_delta_mean_normalizer(self) -> None:
        for wrapper, batch, axis, device, dtype in self._cases():
            with self.subTest(wrapper=wrapper, batch=batch, axis=axis, device=device, dtype=dtype):
                model = self._model(BatchDecoupledVariationalStrategy, wrapper, batch, axis, device, dtype)
                base = model.variational_strategy.base_variational_strategy
                distribution = base._variational_distribution
                constant = 0.5 * distribution.num_inducing_points * math.log(2 * math.pi)
                expected = (self._gaussian_kl(distribution) + constant).sum(axis)
                actual = model.variational_strategy.kl_divergence()
                tolerance = 2e-6 if dtype == torch.float32 else 2e-12
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


if __name__ == "__main__":
    unittest.main()
