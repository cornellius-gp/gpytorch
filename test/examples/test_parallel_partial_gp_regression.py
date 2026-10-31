#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import torch

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import ParallelPartialKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean


# Four sinusoidal functions with noise N(0, 0.1)
def eval_functions(train_x, noisy=True):
    train_y1 = torch.sin(train_x * (2 * pi)) + (torch.randn(train_x.size()) * 0.1 if noisy else 0)
    train_y2 = torch.cos(train_x * (2 * pi)) + (torch.randn(train_x.size()) * 0.1 if noisy else 0)
    train_y3 = torch.cos(train_x * pi) + (torch.randn(train_x.size()) * 0.1 if noisy else 0)
    train_y4 = torch.cos(train_x * pi) + (torch.randn(train_x.size()) * 0.1 if noisy else 0)
    return torch.stack([train_y1, train_y2, train_y3, train_y4], -1)


class ParallelPartialGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ParallelPartialGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=train_y.shape[1])
        self.covar_module = ParallelPartialKernel(RBFKernel(), num_tasks=train_y.shape[1])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestParallelPartialGPRegression(unittest.TestCase):
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

    def test_parallel_partial_gp_mean_abs_error(self):

        # Get training outputs
        train_x = torch.linspace(0, 1, 100)
        train_y = eval_functions(train_x)

        # Likelihood and model
        likelihood = MultitaskGaussianLikelihood(num_tasks=train_y.shape[1])
        model = ParallelPartialGPModel(train_x, train_y, likelihood)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training
        n_iter = 50
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        likelihood.eval()
        test_x = torch.linspace(0, 1, 51)
        test_y = eval_functions(test_x, noisy=False)
        test_preds = likelihood(model(test_x)).mean
        for task in range(train_y.shape[1]):
            mean_abs_error_task = torch.mean(torch.abs(test_y[:, task] - test_preds[:, task]))
            self.assertLess(mean_abs_error_task.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
