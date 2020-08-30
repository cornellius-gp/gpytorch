#!/usr/bin/env python3

import math
import torch
import unittest

import gpytorch
from gpytorch.kernels import RBFKernel, MultitaskKernel, LCMKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.base_kernel_list = [RBFKernel()]
        self.covar_module = LCMKernel(self.base_kernel_list, num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskGPModel_ICM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel_ICM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.base_kernel = RBFKernel()
        self.covar_module = MultitaskKernel(self.base_kernel, num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestLCMKernelRegression(unittest.TestCase):
    def test_lcm_icm_equivalence(self):
        # Training points are every 0.1 in [0,1] (note that they're the same for both tasks)
        train_x = torch.linspace(0, 1, 100)
        # y1 function is sin(2*pi*x) with noise N(0, 0.04)
        train_y1 = torch.sin(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
        # y2 function is cos(2*pi*x) with noise N(0, 0.04)
        train_y2 = torch.cos(train_x.data * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
        # Create a train_y which interleaves the two
        train_y = torch.stack([train_y1, train_y2], -1)

        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood)

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianL^ikelihood parameters
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        n_iter = 50
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        model.eval()
        likelihood.eval()

        # Make predictions for LCM
        with torch.no_grad():
            test_x = torch.linspace(0, 1, 51)
            observed_pred = likelihood(model(test_x))
            mean = observed_pred.mean

        model_icm = MultitaskGPModel_ICM(train_x, train_y, likelihood)
        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        model_icm.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model_icm)
        optimizer = torch.optim.Adam(model_icm.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        for _ in range(n_iter):
            optimizer.zero_grad()
            output = model_icm(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        model_icm.eval()
        likelihood.eval()

        # Make predictions for ICM
        with torch.no_grad():
            test_x = torch.linspace(0, 1, 51)
            observed_pred_icm = likelihood(model_icm(test_x))
            mean_icm = observed_pred_icm.mean

        # Make sure predictions from LCM with one base kernel and ICM are the same.
        self.assertLess((mean - mean_icm).pow(2).mean(), 1e-2)


if __name__ == "__main__":
    unittest.main()
