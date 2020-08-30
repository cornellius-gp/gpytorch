#!/usr/bin/env python3

from math import exp, pi

import os
import random
import torch
import unittest

import gpytorch
from torch import optim, nn
from gpytorch.kernels import RBFKernel, GridInterpolationKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal


# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
def make_data(cuda=False):
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * pi))
    test_x = torch.linspace(0, 1, 51)
    test_y = torch.sin(test_x * (2 * pi))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


data_dim = 1


class SmallFeatureExtractor(nn.Sequential):
    def __init__(self):
        super(SmallFeatureExtractor, self).__init__()
        self.add_module("linear1", nn.Linear(data_dim, 10))
        self.add_module("relu3", nn.ReLU())
        self.add_module("linear4", nn.Linear(10, 1))


feature_extractor = SmallFeatureExtractor()


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)))
        self.covar_module = GridInterpolationKernel(self.base_covar_module, grid_size=50, num_dims=1)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return MultivariateNormal(mean_x, covar_x)


class TestDKLRegression(unittest.TestCase):
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

    def test_dkl_gp_mean_abs_error(self):
        train_x, train_y, test_x, test_y = make_data()
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        with gpytorch.settings.debug(False):
            for _ in range(30):
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

            self.assertLess(mean_abs_error.squeeze().item(), 0.15)

    def test_dkl_gp_fast_pred_var(self):
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            train_x, train_y, test_x, test_y = make_data()
            likelihood = GaussianLikelihood()
            gp_model = GPRegressionModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(30):
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
            # Set the cache
            test_function_predictions = likelihood(gp_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.raw_noise.data.fill_(3)
            test_function_predictions = likelihood(gp_model(train_x))

            noise = likelihood.noise
            var_diff = (test_function_predictions.variance - noise).abs()
            self.assertLess(torch.max(var_diff / noise), 0.15)


if __name__ == "__main__":
    unittest.main()
