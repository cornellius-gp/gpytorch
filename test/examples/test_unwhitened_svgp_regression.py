#!/usr/bin/env python3

import math
import unittest

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.test.base_test_case import BaseTestCase
from torch import optim


def train_data(cuda=False):
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * math.pi))
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class SVGPRegressionModel(ApproximateGP):
    def __init__(self, inducing_points, distribution_cls):
        variational_distribution = distribution_cls(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestSVGPRegression(BaseTestCase, unittest.TestCase):
    seed = 0

    def test_regression_error(
        self,
        cuda=False,
        mll_cls=gpytorch.mlls.VariationalELBO,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
    ):
        train_x, train_y = train_data(cuda=cuda)
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(torch.linspace(0, 1, 25), distribution_cls)
        mll = mll_cls(likelihood, model, num_data=len(train_y))
        if cuda:
            likelihood = likelihood.cuda()
            model = model.cuda()
            mll = mll.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

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


if __name__ == "__main__":
    unittest.main()
