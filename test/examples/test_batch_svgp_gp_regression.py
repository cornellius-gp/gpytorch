#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import optim


def train_data(cuda=False):
    train_x = torch.linspace(0, 1, 260).unsqueeze(-1)
    train_y_cos = torch.cos(train_x * (2 * pi)).squeeze() + 0.01 * torch.randn(260)
    train_y_sin = torch.sin(train_x * (2 * pi)).squeeze() + 0.01 * torch.randn(260)

    # Make train_x (2 x 260 x 1) and train_y (2 x 260)
    train_x = torch.cat([train_x, train_x], dim=1).transpose(-2, 1).unsqueeze(-1)
    train_y = torch.cat([train_y_cos.unsqueeze(-1), train_y_sin.unsqueeze(-1)], dim=1).transpose(-2, -1)
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class SVGPRegressionModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([2])
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
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

    def test_regression_error(self):
        train_x, train_y = train_data()
        likelihood = GaussianLikelihood()
        inducing_points = torch.linspace(0, 1, 25).unsqueeze(-1).repeat(2, 1, 1)
        model = SVGPRegressionModel(inducing_points)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(-1))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)
        for _ in range(180):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss = loss.sum()
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
        mean_abs_error = torch.mean(torch.abs(train_y[0, :] - test_preds[0, :]) / 2)
        mean_abs_error2 = torch.mean(torch.abs(train_y[1, :] - test_preds[1, :]) / 2)
        self.assertLess(mean_abs_error.item(), 1e-1)
        self.assertLess(mean_abs_error2.item(), 1e-1)

    def test_regression_error_shared_inducing_locations(self):
        train_x, train_y = train_data()
        likelihood = GaussianLikelihood()
        inducing_points = torch.linspace(0, 1, 25).unsqueeze(-1)
        model = SVGPRegressionModel(inducing_points)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(-1))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)
        for _ in range(200):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss = loss.sum()
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
        mean_abs_error = torch.mean(torch.abs(train_y[0, :] - test_preds[0, :]) / 2)
        mean_abs_error2 = torch.mean(torch.abs(train_y[1, :] - test_preds[1, :]) / 2)
        self.assertLess(mean_abs_error.item(), 1e-1)
        self.assertLess(mean_abs_error2.item(), 1e-1)

    def test_regression_error_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            train_x, train_y = train_data(cuda=True)
            likelihood = GaussianLikelihood().cuda()
            inducing_points = torch.linspace(0, 1, 25).unsqueeze(-1).repeat(2, 1, 1)
            model = SVGPRegressionModel(inducing_points).cuda()
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(-1))

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()
            optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)
            for _ in range(150):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss = loss.sum()
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
            mean_abs_error = torch.mean(torch.abs(train_y[0, :] - test_preds[0, :]) / 2)
            mean_abs_error2 = torch.mean(torch.abs(train_y[1, :] - test_preds[1, :]) / 2)
            self.assertLess(mean_abs_error.item(), 1e-1)
            self.assertLess(mean_abs_error2.item(), 1e-1)


if __name__ == "__main__":
    unittest.main()
