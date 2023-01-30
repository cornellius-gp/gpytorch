#!/usr/bin/env python3

from math import pi

import torch
import unittest

import gpytorch
from gpytorch.kernels import RBFKernel, GridInterpolationKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.test.base_test_case import BaseTestCase
from torch import optim

# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
n = 40
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_x = train_x
train_y = torch.sin(((train_x[:, 0] + train_x[:, 1]) * (2 * pi)))
train_y = train_y + torch.randn_like(train_y).mul_(0.01)

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_x = test_x
test_y = torch.sin((test_x[:, 0] + test_x[:, 1]) * (2 * pi))
test_y = test_y + torch.randn_like(test_y).mul_(0.01)


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_prior=SmoothedBoxPrior(-1, 1))
        self.base_covar_module = RBFKernel(ard_num_dims=2)
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=16, num_dims=2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestKISSGPKroneckerProductRegression(unittest.TestCase, BaseTestCase):
    seed = 0

    def test_kissgp_gp_mean_abs_error(self):
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        with gpytorch.settings.max_preconditioner_size(5), gpytorch.settings.use_toeplitz(True):
            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(8):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        # Use the other toeplitz option here for testing
        with gpytorch.settings.max_preconditioner_size(5), gpytorch.settings.use_toeplitz(True):
            gp_model.eval()
            likelihood.eval()

            test_preds = likelihood(gp_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
            self.assertLess(mean_abs_error.squeeze().item(), 0.2)

            # Try drawing a sample - make sure there's no errors
            with torch.no_grad(), gpytorch.settings.max_root_decomposition_size(100):
                with gpytorch.settings.fast_pred_samples():
                    gp_model(train_x).rsample(torch.Size([1]))


class TestKISSGPKroneckerProductRegressionDouble(TestKISSGPKroneckerProductRegression):
    def setUp(self):
        super().setUp
        self.default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        super().tearDown()
        torch.set_default_dtype(self.default_dtype)


if __name__ == "__main__":
    unittest.main()
