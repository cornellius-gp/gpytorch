#!/usr/bin/env python3

from math import exp, pi

import os
import random
import torch
import unittest
import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, InducingPointKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal


# Simple training data: let's try to learn a sine function,
# but with SGPR
# let's use 100 training examples.
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


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)))
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=torch.linspace(0, 1, 32), likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestSGPRRegression(unittest.TestCase):
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

    def test_sgpr_mean_abs_error(self):
        train_x, train_y, test_x, test_y = make_data()
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
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

        test_preds = likelihood(gp_model(test_x)).mean
        mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)

    def test_sgpr_fast_pred_var(self):
        train_x, train_y, test_x, test_y = make_data()
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
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

        with gpytorch.settings.max_preconditioner_size(5), gpytorch.settings.max_cg_iterations(50):
            with gpytorch.settings.fast_pred_var(True):
                fast_var = gp_model(test_x).variance
                fast_var_cache = gp_model(test_x).variance
                self.assertLess(torch.max((fast_var_cache - fast_var).abs()), 1e-3)

            with gpytorch.settings.fast_pred_var(False):
                slow_var = gp_model(test_x).variance

        self.assertLess(torch.max((fast_var_cache - slow_var).abs()), 1e-3)

    def test_sgpr_mean_abs_error_cuda(self):
        if torch.cuda.is_available():
            train_x, train_y, test_x, test_y = make_data(cuda=True)
            likelihood = GaussianLikelihood().cuda()
            gp_model = GPRegressionModel(train_x, train_y, likelihood).cuda()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(25):
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
            test_preds = likelihood(gp_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

            self.assertLess(mean_abs_error.squeeze().item(), 0.02)


if __name__ == "__main__":
    unittest.main()
