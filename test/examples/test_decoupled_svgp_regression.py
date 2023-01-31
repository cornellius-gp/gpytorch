#!/usr/bin/env python3

import math
import unittest
from unittest.mock import MagicMock, patch

import linear_operator
import torch
from torch import optim

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.test.base_test_case import BaseTestCase


def train_data():
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * math.pi))
    return train_x, train_y


class SVGPRegressionModel(ApproximateGP):
    def __init__(self, inducing_points, base_inducing_points):
        base_variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            base_inducing_points.size(-1)
        )
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                base_inducing_points,
                base_variational_distribution,
                learn_inducing_locations=True,
                jitter_val=1e-4,
            ),
            inducing_points,
            variational_distribution,
        )
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestSVGPRegression(BaseTestCase, unittest.TestCase):
    seed = 0

    def test_regression_error(
        self,
        mll_cls=gpytorch.mlls.VariationalELBO,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
    ):
        train_x, train_y = train_data()
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(torch.linspace(0, 1, 128), torch.linspace(0, 1, 16))
        mll = mll_cls(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        _wrapped_cg = MagicMock(wraps=linear_operator.utils.linear_cg)
        _cg_mock = patch("linear_operator.utils.linear_cg", new=_wrapped_cg)
        with _cg_mock as cg_mock:
            for _ in range(75):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            for param in model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Set back to eval mode
            model.eval()
            likelihood.eval()
            test_preds = likelihood(model(train_x)).mean.squeeze()
            mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
            self.assertLess(mean_abs_error.item(), 1e-1)

            self.assertFalse(cg_mock.called)

    def test_predictive_ll_regression_error(self):
        return self.test_regression_error(
            mll_cls=gpytorch.mlls.PredictiveLogLikelihood,
            distribution_cls=gpytorch.variational.MeanFieldVariationalDistribution,
        )


if __name__ == "__main__":
    unittest.main()
