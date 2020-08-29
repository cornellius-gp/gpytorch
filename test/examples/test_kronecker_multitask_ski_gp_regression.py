#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import gpytorch
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import GridInterpolationKernel, MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.test.utils import least_used_cuda_device


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self.data_covar_module = GridInterpolationKernel(RBFKernel(), grid_size=100, num_dims=1)
        self.covar_module = MultitaskKernel(self.data_covar_module, num_tasks=2, rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestKroneckerMultiTaskKISSGPRegression(unittest.TestCase):
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

    def _get_data(self, cuda=False):
        # Simple training data: let's try to learn a sine function
        train_x = torch.linspace(0, 1, 100, device=torch.device("cuda") if cuda else torch.device("cpu"))
        # y1 function is sin(2*pi*x) with noise N(0, 0.04)
        train_y1 = torch.sin(train_x * (2 * pi)) + torch.randn_like(train_x) * 0.1
        # y2 function is cos(2*pi*x) with noise N(0, 0.04)
        train_y2 = torch.cos(train_x * (2 * pi)) + torch.randn_like(train_x) * 0.1
        # Create a train_y which interleaves the two
        train_y = torch.stack([train_y1, train_y2], -1)
        return train_x, train_y

    def test_multitask_gp_mean_abs_error(self, cuda=False):
        train_x, train_y = self._get_data(cuda=cuda)
        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood)

        if cuda:
            model.cuda()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        n_iter = 50
        for _ in range(n_iter):
            # Zero prev backpropped gradients
            optimizer.zero_grad()
            # Make predictions from training data
            # Again, note feeding duplicated x_data and indices indicating which task
            output = model(train_x)
            # TODO: Fix this view call!!
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        # Test the model
        model.eval()
        likelihood.eval()

        test_x = torch.linspace(0, 1, 51, device=torch.device("cuda") if cuda else torch.device("cpu"))
        test_y1 = torch.sin(test_x * (2 * pi))
        test_y2 = torch.cos(test_x * (2 * pi))
        test_preds = likelihood(model(test_x)).mean
        mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds[:, 0]))
        mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds[:, 1]))

        self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.05)

    def test_multitask_gp_mean_abs_error_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_multitask_gp_mean_abs_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
