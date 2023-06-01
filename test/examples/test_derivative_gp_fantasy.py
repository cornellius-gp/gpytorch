#!/usr/bin/env python3

import unittest
from math import pi

import torch

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernelGrad
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMeanGrad
from gpytorch.test.base_test_case import BaseTestCase

# Simple training data
num_train_samples = 15
num_fantasies = 10
dim = 1
train_X = torch.linspace(0, 1, num_train_samples).reshape(-1, 1)
train_Y = torch.hstack([
    torch.sin(train_X * (2 * pi)).reshape(-1, 1),
    (2 * pi) * torch.cos(train_X * (2 * pi)).reshape(-1, 1),
])


class GPWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_Y):
        likelihood = MultitaskGaussianLikelihood(num_tasks=1 + dim)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = ConstantMeanGrad()
        self.base_kernel = RBFKernelGrad()
        self.covar_module = ScaleKernel(self.base_kernel)
        self._num_outputs = 1 + dim

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestDerivativeGPFutures(BaseTestCase, unittest.TestCase):

    # Inspired by test_lanczos_fantasy_model
    def test_derivative_gp_futures(self):
        model = GPWithDerivatives(train_X, train_Y)
        mll = gpytorch.mlls.sum_marginal_log_likelihood.ExactMarginalLogLikelihood(model.likelihood, model)

        mll.train()
        mll.eval()

        # get a posterior to fill in caches
        model(torch.randn(num_train_samples).reshape(-1, 1))

        new_x = torch.randn((1, 1, dim))
        new_y = torch.randn((num_fantasies, 1, 1, 1 + dim))

        # just check that this can run without error
        model.get_fantasy_model(new_x, new_y)


if __name__ == "__main__":
    unittest.main()
