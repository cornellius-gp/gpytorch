import unittest

import torch

from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, Likelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ExactGP
from gpytorch.test.base_test_case import BaseTestCase


class SingleGPModel(ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(SingleGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(ScaleKernel(RBFKernel()), num_tasks=num_tasks)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestMissingData(BaseTestCase, unittest.TestCase):
    seed = 1

    def _train(self, model: ExactGP, likelihood: Likelihood):
        model.train()
        likelihood.train()

        mll = ExactMarginalLogLikelihood(likelihood, model, nan_means_missing_data=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.15)

        for _ in range(20):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = mll(output, model.train_targets)
            self.assertFalse(torch.any(torch.isnan(output.mean)).item())
            self.assertFalse(torch.any(torch.isnan(output.covariance_matrix)).item())
            self.assertFalse(torch.isnan(loss).item())
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

    def test_single(self):
        train_x = torch.linspace(0, 1, 21)
        test_x = torch.linspace(0, 1, 51)
        train_y = torch.sin(2 * torch.pi * train_x)
        train_y += torch.normal(0, 0.01, train_y.shape)
        train_y[::2] = torch.nan

        likelihood = GaussianLikelihood()
        model = SingleGPModel(train_x, train_y, likelihood)
        self._train(model, likelihood)

        with torch.no_grad():
            prediction = model(test_x)

        self.assertFalse(torch.any(torch.isnan(prediction.mean)).item())
        self.assertFalse(torch.any(torch.isnan(prediction.covariance_matrix)).item())

    def test_multitask(self):
        num_tasks = 10
        train_x = torch.linspace(0, 1, 21)
        test_x = torch.linspace(0, 1, 51)
        train_y = torch.sin(2 * torch.pi * train_x)[:, None] * torch.rand(1, num_tasks)
        train_y += torch.normal(0, 0.01, train_y.shape)
        train_y[::3, : num_tasks // 2] = torch.nan
        train_y[::4, num_tasks // 2 :] = torch.nan

        likelihood = MultitaskGaussianLikelihood(num_tasks)
        model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks)
        self._train(model, likelihood)

        with torch.no_grad():
            prediction = model(test_x)

        self.assertFalse(torch.any(torch.isnan(prediction.mean)).item())
        self.assertFalse(torch.any(torch.isnan(prediction.covariance_matrix)).item())
