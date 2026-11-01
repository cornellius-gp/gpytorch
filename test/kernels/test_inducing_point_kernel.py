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


class TestInducingPointKernel(unittest.TestCase):
    def test_dtype_conversion_clears_kernel_cache(self):
        x = torch.tensor([[0.2], [0.8]])
        inducing_points = torch.tensor([[0.0], [0.5], [1.0]])
        for source_dtype, target_dtype in ((torch.float32, torch.float64), (torch.float64, torch.float32)):
            with self.subTest(source_dtype=source_dtype), torch.no_grad():
                kernel = InducingPointKernel(
                    RBFKernel(), inducing_points.clone(), gpytorch.likelihoods.GaussianLikelihood()
                )
                kernel.to(dtype=source_dtype).eval()
                kernel(x.to(dtype=source_dtype)).to_dense()
                kernel.to(dtype=target_dtype)
                reference = InducingPointKernel(
                    RBFKernel(), inducing_points.clone(), gpytorch.likelihoods.GaussianLikelihood()
                )
                reference.to(dtype=target_dtype).eval()
                reference.load_state_dict(kernel.state_dict())
                actual = kernel(x.to(dtype=target_dtype)).to_dense()
                expected = reference(x.to(dtype=target_dtype)).to_dense()
                self.assertEqual(actual.dtype, target_dtype)
                torch.testing.assert_close(actual, expected)

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
