#!/usr/bin/env python3

import math
import warnings
import unittest
from unittest.mock import MagicMock, patch

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import AbstractVariationalGP
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.lazy import ExtraComputationWarning
from torch import optim


def train_data(cuda=False):
    train_x = torch.linspace(0, 1, 260)
    train_y = torch.cos(train_x * (2 * math.pi))
    if cuda:
        return train_x.cuda(), train_y.cuda()
    else:
        return train_x, train_y


class SVGPRegressionModel(AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(-1))
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


class TestSVGPRegression(BaseTestCase, unittest.TestCase):
    seed = 0

    def test_regression_error(self, cuda=False):
        train_x, train_y = train_data(cuda=cuda)
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(torch.linspace(0, 1, 25))
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
        if cuda:
            likelihood = likelihood.cuda()
            model = model.cuda()
            mll = mll.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)
        with warnings.catch_warnings(record=True) as ws, _cg_mock as cg_mock:
            for _ in range(150):
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

            # Make sure CG was called (or not), and no warnings were thrown
            self.assertFalse(cg_mock.called)
            self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))

    def test_regression_error_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            return self.test_regression_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
