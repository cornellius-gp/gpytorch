#!/usr/bin/env python3

from math import pi

import os
import random
import torch
import unittest
import gpytorch
from torch import optim
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy


def train_data(cuda=False):
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * pi))
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class SVGPRegressionModel(AbstractVariationalGP):
    def __init__(self, inducing_points, learn_locs=True):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = WhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=learn_locs
        )
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(0.001, 1.0, sigma=0.1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestSVGPRegression(unittest.TestCase):
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

    def test_regression_error(self, skip_logdet_forward=False, cuda=False):
        train_x, train_y = train_data(cuda=cuda)
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(torch.linspace(0, 1, 25))
        if cuda:
            likelihood.cuda()
            model.cuda()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)
        with gpytorch.settings.skip_logdet_forward(skip_logdet_forward):
            for _ in range(170):
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

    def test_regression_error_cuda(self):
        if torch.cuda.is_available():
            self.test_regression_error(skip_logdet_forward=False, cuda=True)

    def test_regression_error_full(self, skip_logdet_forward=False, cuda=False):
        train_x, train_y = train_data(cuda=cuda)
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(inducing_points=train_x, learn_locs=False)
        if cuda:
            likelihood.cuda()
            model.cuda()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)
        with gpytorch.settings.skip_logdet_forward(skip_logdet_forward):
            for _ in range(200):
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

    def test_regression_error_full_cuda(self):
        if torch.cuda.is_available():
            self.test_regression_error_full(skip_logdet_forward=False, cuda=True)

    def test_regression_error_skip_logdet_forward(self):
        self.test_regression_error(skip_logdet_forward=True)

    def test_regression_error_skip_logdet_forward_cuda(self):
        if torch.cuda.is_available():
            self.test_regression_error(skip_logdet_forward=True, cuda=True)


if __name__ == "__main__":
    unittest.main()
