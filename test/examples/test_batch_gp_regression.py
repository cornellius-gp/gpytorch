#!/usr/bin/env python3

import math
import os
import random
import unittest

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from torch import optim


# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x1 = torch.linspace(0, 2, 11).unsqueeze(-1)
train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
train_y1.add_(torch.randn_like(train_y1).mul_(0.01))
test_x1 = torch.linspace(0, 2, 51).unsqueeze(-1)
test_y1 = torch.sin(test_x1 * (2 * math.pi)).squeeze()

train_x2 = torch.linspace(0, 1, 11).unsqueeze(-1)
train_y2 = torch.sin(train_x2 * (2 * math.pi)).squeeze()
train_y2.add_(torch.randn_like(train_y2).mul_(0.01))
test_x2 = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y2 = torch.sin(test_x2 * (2 * math.pi)).squeeze()

# Combined sets of data
train_x12 = torch.cat((train_x1.unsqueeze(0), train_x2.unsqueeze(0)), dim=0).contiguous()
train_y12 = torch.cat((train_y1.unsqueeze(0), train_y2.unsqueeze(0)), dim=0).contiguous()
test_x12 = torch.cat((test_x1.unsqueeze(0), test_x2.unsqueeze(0)), dim=0).contiguous()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape=torch.Size()):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(batch_shape=batch_shape, prior=gpytorch.priors.SmoothedBoxPrior(-1, 1))
        self.covar_module = ScaleKernel(
            RBFKernel(
                batch_shape=batch_shape,
                lengthscale_prior=gpytorch.priors.NormalPrior(
                    loc=torch.zeros(*batch_shape, 1, 1), scale=torch.ones(*batch_shape, 1, 1)
                ),
            ),
            batch_shape=batch_shape,
            outputscale_prior=gpytorch.priors.SmoothedBoxPrior(-2, 2),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestBatchGPRegression(unittest.TestCase):
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

    def test_train_on_single_set_test_on_batch(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood()
        gp_model = ExactGPModel(train_x1, train_y1, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(75):
            optimizer.zero_grad()
            output = gp_model(train_x1)
            loss = -mll(output, train_y1).sum()
            loss.backward()
            optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        # First test on non-batch
        non_batch_predictions = likelihood(gp_model(test_x1))
        preds1 = non_batch_predictions.mean
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)

        # Make predictions for both sets of test points, and check MAEs.
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.1)

        # Smoke test for non-batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x1.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)

        # Smoke test for batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x12.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)

    def test_train_on_batch_test_on_batch(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(
            noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(2), scale=torch.ones(2)),
            batch_shape=torch.Size([2]),
        )
        gp_model = ExactGPModel(train_x12, train_y12, likelihood, batch_shape=torch.Size([2]))
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x12)
            loss = -mll(output, train_y12, train_x12).sum()
            loss.backward()
            optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        # First test on non-batch
        non_batch_predictions = likelihood(gp_model(test_x1))
        preds1 = non_batch_predictions.mean
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1[0]))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)

        # Make predictions for both sets of test points, and check MAEs.
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.1)

        # Smoke test for batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x12.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)

        # Smoke test for non-batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x1.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)

    def test_train_on_batch_test_on_batch_shared_hypers_over_batch(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(
            noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(2), scale=torch.ones(2))
        )
        gp_model = ExactGPModel(train_x12, train_y12, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x12)
            loss = -mll(output, train_y12, train_x12).sum()
            loss.backward()
            optimizer.step()

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        gp_model.eval()
        likelihood.eval()

        # First test on non-batch
        non_batch_predictions = likelihood(gp_model(test_x1))
        preds1 = non_batch_predictions.mean
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1[0]))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)

        # Make predictions for both sets of test points, and check MAEs.
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.1)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.1)

        # Smoke test for batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x12.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)

        # Smoke test for non-batch mode derivatives failing
        test_x_param = torch.nn.Parameter(test_x1.data)
        batch_predictions = likelihood(gp_model(test_x_param))
        batch_predictions.mean.sum().backward()
        self.assertTrue(test_x_param.grad is not None)


if __name__ == "__main__":
    unittest.main()
