#!/usr/bin/env python3

import os
import random
import unittest
from math import pi

import torch

import gpytorch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import MultitaskKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.utils.memoize import get_from_cache


# Simple training data: let's try to learn a sine function
train_x = torch.linspace(0, 1, 100)

# y1 function is sin(2*pi*x) with noise N(0, 0.04)
train_y1 = torch.sin(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.1
# y2 function is cos(2*pi*x) with noise N(0, 0.04)
train_y2 = torch.cos(train_x * (2 * pi)) + torch.randn(train_x.size()) * 0.1

# Create a train_y which interleaves the two
train_y = torch.stack([train_y1, train_y2], -1)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=2)
        self_covar_module = RBFKernel()
        self.covar_module = MultitaskKernel(self_covar_module, num_tasks=2, rank=2)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestKroneckerMultiTaskGPRegression(unittest.TestCase):
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

    def test_multitask_gp_mean_abs_error(self):
        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        model = MultitaskGPModel(train_x, train_y, likelihood)
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
        test_x = torch.linspace(0, 1, 51)
        test_y1 = torch.sin(test_x * (2 * pi))
        test_y2 = torch.cos(test_x * (2 * pi))
        test_preds = likelihood(model(test_x)).mean
        mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds[:, 0]))
        mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds[:, 1]))

        self.assertLess(mean_abs_error_task_1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error_task_2.squeeze().item(), 0.05)

    def _get_fantasy_data(self):
        # A well spaced subset of the data, so that the kernel matrix of the untrained model is well conditioned
        sub_x, sub_y = train_x[::5], train_y[::5]
        return sub_x[:15], sub_y[:15], sub_x[15:], sub_y[15:], sub_x, sub_y

    def _get_full_model(self, model, full_x, full_y):
        # The same model, but conditioned on the fantasy points from the start
        full_model = MultitaskGPModel(full_x, full_y, MultitaskGaussianLikelihood(num_tasks=2))
        full_model.load_state_dict(model.state_dict())
        full_model.eval()
        return full_model

    def test_multitask_fantasy_model(self):
        # Fantasy updates used to fail for multitask models whenever more than one point was added
        obs_x, obs_y, fant_x, fant_y, full_x, full_y = self._get_fantasy_data()
        test_x = torch.linspace(0, 1, 11)

        model = MultitaskGPModel(obs_x, obs_y, MultitaskGaussianLikelihood(num_tasks=2))
        model.eval()
        model(test_x)  # Fill in the test time caches
        fant_model = model.get_fantasy_model(fant_x, fant_y)

        # A fantasy update has to give the same posterior as conditioning on all of the data at once
        full_model = self._get_full_model(model, full_x, full_y)
        fant_pred, full_pred = fant_model(test_x), full_model(test_x)
        self.assertEqual(fant_pred.mean.shape, full_pred.mean.shape)
        self.assertTrue(torch.allclose(fant_pred.mean, full_pred.mean, atol=1e-4))
        self.assertTrue(torch.allclose(fant_pred.covariance_matrix, full_pred.covariance_matrix, atol=1e-4))

        # Check the incrementally updated mean cache - which is what the fantasy strategy computes -
        # against the one obtained by solving the full system from scratch
        self.assertTrue(
            torch.allclose(
                get_from_cache(fant_model.prediction_strategy, "mean_cache"),
                full_model.prediction_strategy.mean_cache,
                atol=1e-4,
            )
        )

    def test_multitask_fantasy_model_batch_targets(self):
        # The fantasy targets may have an extra batch dimension (f x m x t) with shared fantasy inputs
        obs_x, obs_y, fant_x, fant_y, full_x, full_y = self._get_fantasy_data()
        test_x = torch.linspace(0, 1, 11)
        num_fantasies = 3

        model = MultitaskGPModel(obs_x, obs_y, MultitaskGaussianLikelihood(num_tasks=2))
        model.eval()
        model(test_x)  # Fill in the test time caches
        fant_model = model.get_fantasy_model(fant_x, fant_y.unsqueeze(0).repeat(num_fantasies, 1, 1))

        full_model = self._get_full_model(model, full_x, full_y)
        fant_pred, full_pred = fant_model(test_x), full_model(test_x)
        self.assertEqual(fant_pred.mean.shape, torch.Size([num_fantasies, *full_pred.mean.shape]))
        for fantasy_mean in fant_pred.mean:
            self.assertTrue(torch.allclose(fantasy_mean, full_pred.mean, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
