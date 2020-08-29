#!/usr/bin/env python3

import random
import os
import torch
import unittest

import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, AdditiveStructureKernel, GridInterpolationKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal

n = 20
train_x = torch.zeros(pow(n, 2), 2)
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
train_y = torch.sin(train_x[:, 0]) + torch.cos(train_x[:, 1])
train_y = train_y + torch.randn_like(train_y).div_(20.0)

m = 10
test_x = torch.zeros(pow(m, 2), 2)
for i in range(m):
    for j in range(m):
        test_x[i * m + j][0] = float(i) / (m - 1)
        test_x[i * m + j][1] = float(j) / (m - 1)
test_y = torch.sin(test_x[:, 0]) + torch.cos(test_x[:, 1])


# All tests that pass with the exact kernel should pass with the interpolated kernel.
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.base_covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))
        self.covar_module = AdditiveStructureKernel(
            GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1), num_dims=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestKISSGPAdditiveRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_kissgp_gp_mean_abs_error(self, toeplitz=False):
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        with gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.max_cg_iterations(50):
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.use_toeplitz(toeplitz):
                # Optimize the model
                gp_model.train()
                likelihood.train()

                optimizer = optim.Adam(gp_model.parameters(), lr=0.01)
                optimizer.n_iter = 0
                for _ in range(15):
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
                gp_model.eval()
                likelihood.eval()

                test_preds = likelihood(gp_model(test_x)).mean
                mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
                self.assertLess(mean_abs_error.squeeze().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
