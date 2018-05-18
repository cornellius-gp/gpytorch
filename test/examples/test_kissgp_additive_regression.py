from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
import torch
import unittest
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel, AdditiveGridInterpolationKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
n = 40
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_x = Variable(train_x)
train_y = Variable(
    (torch.sin(train_x.data[:, 0]) + torch.cos(train_x.data[:, 1])) * (2 * math.pi)
)

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_x = Variable(test_x)
test_y = Variable(
    (torch.sin(test_x.data[:, 0]) + torch.cos(test_x.data[:, 1])) * (2 * math.pi)
)


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class GPRegressionModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(constant_bounds=(-1, 1))
        self.base_covar_module = RBFKernel(log_lengthscale_bounds=(-3, 3))
        self.covar_module = AdditiveGridInterpolationKernel(
            self.base_covar_module, grid_size=100, grid_bounds=[(0, 1)], n_components=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class TestKISSGPAdditiveRegression(unittest.TestCase):

    def setUp(self):
        if (
            os.getenv("UNLOCK_SEED") is None
            or os.getenv("UNLOCK_SEED").lower() == "false"
        ):
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_kissgp_gp_mean_abs_error(self):
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x.data, train_y.data, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        with gpytorch.settings.max_preconditioner_size(
            10
        ), gpytorch.settings.max_cg_iterations(
            30
        ):
            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(
                list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.2
            )
            optimizer.n_iter = 0
            for _ in range(20):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            gp_model.eval()
            likelihood.eval()

            test_preds = likelihood(gp_model(test_x)).mean()
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
            self.assertLess(mean_abs_error.data.squeeze().item(), 0.15)


if __name__ == "__main__":
    unittest.main()
