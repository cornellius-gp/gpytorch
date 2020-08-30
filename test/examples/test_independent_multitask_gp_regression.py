#!/usr/bin/env python3

import math
import os
import random
import unittest

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from torch import optim


# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x = torch.linspace(0, 2, 51).unsqueeze(-1)
test_x = torch.linspace(0, 2, 11).unsqueeze(-1)
train_y1 = torch.sin(train_x * (2 * math.pi)).squeeze()
train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
test_y1 = torch.sin(test_x * (2 * math.pi)).squeeze()
train_y2 = torch.sin(train_x * (2 * math.pi)).squeeze()
train_y2.add_(torch.randn_like(train_y2).mul_(0.01))
test_y2 = torch.sin(test_x * (2 * math.pi)).squeeze()

# Combined sets of data
train_y12 = torch.stack((train_y1, train_y2), dim=-1).contiguous()
test_y12 = torch.stack((test_y1, test_y2), dim=-1).contiguous()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape=torch.Size([2])):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal.from_batch_mvn(MultivariateNormal(mean_x, covar_x))


class TestIndependentMultitaskGPRegression(unittest.TestCase):
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

    def test_train_and_eval(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = MultitaskGaussianLikelihood(num_tasks=2)
        gp_model = ExactGPModel(train_x, train_y12, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(75):
            optimizer.zero_grad()
            output = gp_model(train_x)
            loss = -mll(output, train_y12).sum()
            loss.backward()
            optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        # Make predictions for both sets of test points, and check MAEs.
        with torch.no_grad(), gpytorch.settings.max_eager_kernel_size(1):
            batch_predictions = likelihood(gp_model(test_x))
            preds1 = batch_predictions.mean[:, 0]
            preds2 = batch_predictions.mean[:, 1]
            mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
            mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
            self.assertLess(mean_abs_error1.squeeze().item(), 0.01)
            self.assertLess(mean_abs_error2.squeeze().item(), 0.01)

            # Smoke test for getting predictive uncertainties
            lower, upper = batch_predictions.confidence_region()
            self.assertEqual(lower.shape, test_y12.shape)
            self.assertEqual(upper.shape, test_y12.shape)


if __name__ == "__main__":
    unittest.main()
