#!/usr/bin/env python3

import os
import random
import math
import torch
import unittest

import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, MultitaskKernel
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultitaskMultivariateNormal


# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x1 = torch.linspace(0, 1, 11).unsqueeze(-1)
train_y1 = torch.cat([torch.sin(train_x1 * (2 * math.pi)), torch.cos(train_x1 * (2 * math.pi))], 1)
test_x1 = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y1 = torch.cat([torch.sin(test_x1 * (2 * math.pi)), torch.cos(test_x1 * (2 * math.pi))], 1)

train_x2 = torch.linspace(0, 1, 11).unsqueeze(-1)
train_y2 = torch.cat([torch.sin(train_x2 * (2 * math.pi)), torch.cos(train_x2 * (2 * math.pi))], 1)
test_x2 = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y2 = torch.cat([torch.sin(test_x2 * (2 * math.pi)), torch.cos(test_x2 * (2 * math.pi))], 1)

# Combined sets of data
train_x12 = torch.cat((train_x1.unsqueeze(0), train_x2.unsqueeze(0)), dim=0).contiguous()
train_y12 = torch.cat((train_y1.unsqueeze(0), train_y2.unsqueeze(0)), dim=0).contiguous()
test_x12 = torch.cat((test_x1.unsqueeze(0), test_x2.unsqueeze(0)), dim=0).contiguous()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, batch_shape=torch.Size()):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = MultitaskMean(
            ConstantMean(batch_shape=batch_shape, prior=gpytorch.priors.SmoothedBoxPrior(-1, 1)), num_tasks=2
        )
        self.covar_module = MultitaskKernel(
            RBFKernel(
                batch_shape=batch_shape,
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=torch.tensor(0.0), scale=torch.tensor(1.0)),
            ),
            num_tasks=2,
            rank=1,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


class TestBatchMultitaskGPRegression(unittest.TestCase):
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
        likelihood = MultitaskGaussianLikelihood(
            noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(1), scale=torch.ones(1)), num_tasks=2
        )
        gp_model = ExactGPModel(train_x1, train_y1, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
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

        # Make predictions for both sets of test points, and check MAEs.
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.05)

    def test_train_on_batch_test_on_batch(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = MultitaskGaussianLikelihood(
            noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(2), scale=torch.ones(2)),
            batch_shape=torch.Size([2]),
            num_tasks=2,
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
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.05)

    def test_train_on_batch_test_on_batch_shared_hypers_over_batch(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = MultitaskGaussianLikelihood(
            noise_prior=gpytorch.priors.NormalPrior(loc=torch.zeros(2), scale=torch.ones(2)),
            batch_shape=torch.Size(),
            num_tasks=2,
        )
        gp_model = ExactGPModel(train_x12, train_y12, likelihood, batch_shape=torch.Size())
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x12)
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
        batch_predictions = likelihood(gp_model(test_x12))
        preds1 = batch_predictions.mean[0]
        preds2 = batch_predictions.mean[1]
        mean_abs_error1 = torch.mean(torch.abs(test_y1 - preds1))
        mean_abs_error2 = torch.mean(torch.abs(test_y2 - preds2))
        self.assertLess(mean_abs_error1.squeeze().item(), 0.05)
        self.assertLess(mean_abs_error2.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
