#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, patch

import linear_operator
import torch

import gpytorch
from gpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel


class _TestModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = InducingPointKernel(
            ScaleKernel(RBFKernel(ard_num_dims=3)),
            inducing_points=torch.randn(512, 3),
            likelihood=likelihood,
        )

    def forward(self, input):
        mean = self.mean_module(input)
        covar = self.covar_module(input)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class _BatchIndependentInducingPointModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, inducing_points):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.size(-1))
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.size(-1)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_tasks]))
        base_covar_module = ScaleKernel(
            RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks]),
        )
        self.covar_module = InducingPointKernel(
            base_covar_module,
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class TestInducingPointKernel(unittest.TestCase):
    def test_kernel_output(self):
        train_x = torch.randn(1000, 3)
        train_y = torch.randn(1000)
        test_x = torch.randn(500, 3)
        model = _TestModel(train_x, train_y)

        # Make sure that the prior kernel is the correct type
        model.train()
        output = model(train_x).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, linear_operator.operators.LowRankRootLinearOperator)

        # Make sure that the prior predictive kernel is the correct type
        model.train()
        output = model.likelihood(model(train_x)).lazy_covariance_matrix.evaluate_kernel()
        self.assertIsInstance(output, linear_operator.operators.LowRankRootAddedDiagLinearOperator)

        # Make sure we're calling the correct prediction strategy
        _wrapped_ps = MagicMock(wraps=gpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy)
        with patch("gpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy", new=_wrapped_ps) as ps_mock:
            model.eval()
            output = model.likelihood(model(test_x))
            _ = output.mean + output.variance  # Compute something to break through any lazy evaluations
            self.assertTrue(ps_mock.called)

        # Check whether changing diagonal correction makes a difference (ensuring that cache is cleared)
        model.train()
        model.eval()
        with gpytorch.settings.sgpr_diagonal_correction(True), torch.no_grad():
            output_mean_correct = model(test_x).mean
        model.train()
        model.eval()
        with gpytorch.settings.sgpr_diagonal_correction(False), torch.no_grad():
            output_mean_no_correct = model(test_x).mean
        self.assertNotAlmostEqual(output_mean_correct.sum().item(), output_mean_no_correct.sum().item())

    def test_batch_independent_multitask_uses_sgpr_prediction_strategy(self):
        # Regression for https://github.com/cornellius-gp/gpytorch/issues/2659
        nx, ny = 2, 3
        train_x = torch.rand(100, nx)
        train_y = torch.rand(100, ny)
        test_x = torch.rand(20, nx)
        inducing_points = torch.rand(ny, 10, nx)

        model = _BatchIndependentInducingPointModel(train_x, train_y, inducing_points)
        model.eval()
        model.likelihood.eval()

        with torch.no_grad():
            pred = model(test_x)

        self.assertIsInstance(
            model.prediction_strategy,
            gpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy,
        )
        self.assertEqual(pred.mean.shape, torch.Size([20, ny]))
        self.assertEqual(pred.variance.shape, torch.Size([20, ny]))
