#!/usr/bin/env python3

import math
import unittest

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList
from gpytorch.means import ConstantMean
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.models import IndependentModelList
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.test.utils import least_used_cuda_device


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = ScaleKernel(
            RBFKernel(lengthscale_prior=SmoothedBoxPrior(math.exp(-3), math.exp(3), sigma=0.1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestModelListGPRegression(unittest.TestCase):
    def test_simple_model_list_gp_regression(self, cuda=False):
        train_x1 = torch.linspace(0, 0.95, 25) + 0.05 * torch.rand(25)
        train_x2 = torch.linspace(0, 0.95, 15) + 0.05 * torch.rand(15)

        train_y1 = torch.sin(train_x1 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x1)
        train_y2 = torch.cos(train_x2 * (2 * math.pi)) + 0.2 * torch.randn_like(train_x2)

        likelihood1 = GaussianLikelihood()
        model1 = ExactGPModel(train_x1, train_y1, likelihood1)

        likelihood2 = GaussianLikelihood()
        model2 = ExactGPModel(train_x2, train_y2, likelihood2)

        model = IndependentModelList(model1, model2)
        likelihood = LikelihoodList(model1.likelihood, model2.likelihood)

        if cuda:
            model = model.cuda()

        model.train()
        likelihood.train()

        mll = SumMarginalLogLikelihood(likelihood, model)

        optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.linspace(0, 1, 10, device=torch.device("cuda") if cuda else torch.device("cpu"))
            outputs_f = model(test_x, test_x)
            predictions_obs_noise = likelihood(*outputs_f)

        self.assertIsInstance(outputs_f, list)
        self.assertEqual(len(outputs_f), 2)
        self.assertIsInstance(predictions_obs_noise, list)
        self.assertEqual(len(predictions_obs_noise), 2)

    def test_simple_model_list_gp_regression_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_simple_model_list_gp_regression(cuda=True)


if __name__ == "__main__":
    unittest.main()
