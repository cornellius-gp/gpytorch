#!/usr/bin/env python3

import os
import random
import unittest
import warnings
from math import pi

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RFFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.utils.warnings import NumericalWarning
from torch import optim


# Simple training data: let's try to learn a sine function,
# let's use 100 training examples.
def make_data(cuda=False):
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * pi))
    train_y.add_(torch.randn_like(train_y), alpha=1e-2)
    test_x = torch.rand(51)
    test_y = torch.sin(test_x * (2 * pi))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.covar_module = ScaleKernel(RFFKernel(num_samples=10))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestRFFRegression(unittest.TestCase):
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

    def test_rff_mean_abs_error(self):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        train_x, train_y, test_x, test_y = make_data()
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(30):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            # Check that we have the right LazyTensor type
            kernel = likelihood(gp_model(train_x)).lazy_covariance_matrix.evaluate_kernel()
            self.assertIsInstance(kernel, gpytorch.lazy.LowRankRootAddedDiagLazyTensor)

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        test_preds = likelihood(gp_model(test_x)).mean
        mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)
