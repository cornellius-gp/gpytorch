#!/usr/bin/env python3

import math
import warnings
import unittest
from unittest.mock import MagicMock, patch
from test._utils import least_used_cuda_device

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import optim
from test._base_test_case import BaseTestCase


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

    def test_regression_error(self, cuda=False, skip_logdet_forward=False, cholesky=False):
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
        with gpytorch.settings.max_cholesky_size(math.inf if cholesky else 0), \
                gpytorch.settings.skip_logdet_forward(skip_logdet_forward), \
                warnings.catch_warnings(record=True) as w, \
                patch("gpytorch.utils.linear_cg", new=_wrapped_cg) as linear_cg_mock:
            for _ in range(150):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

            # Make sure CG was called (or not), and no warnings were thrown
            self.assertEqual(len(w), 0)
            if cholesky:
                self.assertFalse(linear_cg_mock.called)
            else:
                self.assertTrue(linear_cg_mock.called)

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

    def test_regression_error_skip_logdet_forward(self):
        return self.test_regression_error(skip_logdet_forward=True)

    def test_regression_error_cholesky(self):
        return self.test_regression_error(cholesky=True)

    def test_regression_error_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            return self.test_regression_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
