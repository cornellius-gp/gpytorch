#!/usr/bin/env python3

import warnings

import math
import os
import random
import unittest

import gpytorch
import torch
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.warnings import GPInputWarning
from torch import optim


def make_data(grid, cuda=False):
    train_x = gpytorch.utils.grid.create_data_from_grid(grid)
    train_y = torch.sin((train_x.sum(-1)) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
    n = 20
    test_x = torch.zeros(int(pow(n, 2)), 2)
    for i in range(n):
        for j in range(n):
            test_x[i * n + j][0] = float(i) / (n - 1)
            test_x[i * n + j][1] = float(j) / (n - 1)
    test_y = torch.sin(((test_x.sum(-1)) * (2 * math.pi)))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class GridGPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, grid, train_x, train_y, likelihood):
        super(GridGPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridKernel(gpytorch.kernels.RBFKernel(), grid=grid)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestGridGPRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_grid_gp_mean_abs_error(self, num_dim=1, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        grid_bounds = [(0, 1)] if num_dim == 1 else [(0, 1), (0, 2)]
        grid_size = 25
        grid = torch.zeros(grid_size, len(grid_bounds), device=device)
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(
                grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size, device=device
            )

        train_x, train_y, test_x, test_y = make_data(grid, cuda=cuda)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp_model = GridGPRegressionModel(grid, train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        with gpytorch.settings.debug(True):
            for _ in range(20):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for name, param in gp_model.named_parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            gp_model.eval()
            likelihood.eval()
            # Make sure we don't get GP input warnings for testing on training data
            warnings.simplefilter("ignore", GPInputWarning)

            train_preds = likelihood(gp_model(train_x)).mean
            mean_abs_error = torch.mean(torch.abs(train_y - train_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.3)

    def test_grid_gp_mean_abs_error_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_grid_gp_mean_abs_error(cuda=True)

    def test_grid_gp_mean_abs_error_2d(self):
        self.test_grid_gp_mean_abs_error(num_dim=2)

    def test_grid_gp_mean_abs_error_2d_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_grid_gp_mean_abs_error(cuda=True, num_dim=2)


if __name__ == "__main__":
    unittest.main()
