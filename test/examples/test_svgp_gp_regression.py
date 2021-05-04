#!/usr/bin/env python3

import math
import warnings
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import gpytorch
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.warnings import OldVersionWarning
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
        variational_strategy = gpytorch.variational.VariationalStrategy(
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

    def test_loading_old_model(self):
        train_x, train_y = train_data(cuda=False)
        likelihood = GaussianLikelihood()
        model = SVGPRegressionModel(torch.linspace(0, 1, 25), gpytorch.variational.CholeskyVariationalDistribution)
        data_file = Path(__file__).parent.joinpath("old_variational_strategy_model.pth").resolve()
        state_dicts = torch.load(data_file)
        likelihood.load_state_dict(state_dicts["likelihood"], strict=False)

        # Ensure we get a warning
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always", OldVersionWarning)

            model.load_state_dict(state_dicts["model"])
            self.assertTrue(any(issubclass(w.category, OldVersionWarning) for w in ws))

        with torch.no_grad():
            model.eval()
            likelihood.eval()
            test_preds = likelihood(model(train_x)).mean.squeeze()
            mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
            self.assertLess(mean_abs_error.item(), 1e-1)

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

        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)
        with _cg_mock as cg_mock:
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

    def test_predictive_ll_regression_error(self):
        return self.test_regression_error(
            mll_cls=gpytorch.mlls.PredictiveLogLikelihood,
            distribution_cls=gpytorch.variational.MeanFieldVariationalDistribution,
        )

    def test_predictive_ll_regression_error_delta(self):
        return self.test_regression_error(
            mll_cls=gpytorch.mlls.PredictiveLogLikelihood,
            distribution_cls=gpytorch.variational.DeltaVariationalDistribution,
        )

    def test_robust_regression_error(self):
        return self.test_regression_error(mll_cls=gpytorch.mlls.GammaRobustVariationalELBO)

    def test_regression_error_cuda(self):
        if not torch.cuda.is_available():
            return
        with least_used_cuda_device():
            return self.test_regression_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
