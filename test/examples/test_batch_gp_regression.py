from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import math
import torch
import unittest
import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x1 = torch.linspace(0, 1, 11).unsqueeze(-1)
train_y1 = torch.sin(train_x1 * (2 * math.pi)).squeeze()
test_x1 = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y1 = torch.sin(test_x1 * (2 * math.pi)).squeeze()

train_x2 = torch.linspace(0, 1, 11).unsqueeze(-1)
train_y2 = torch.cos(train_x2 * (2 * math.pi)).squeeze()
test_x2 = torch.linspace(0, 1, 51).unsqueeze(-1)
test_y2 = torch.cos(test_x2 * (2 * math.pi)).squeeze()

# Combined sets of data
train_x12 = torch.cat((train_x1.unsqueeze(0), train_x2.unsqueeze(0)), dim=0).contiguous()
train_y12 = torch.cat((train_y1.unsqueeze(0), train_y2.unsqueeze(0)), dim=0).contiguous()
test_x12 = torch.cat((test_x1.unsqueeze(0), test_x2.unsqueeze(0)), dim=0).contiguous()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, batch_size=1):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(batch_size=batch_size)
        self.covar_module = ScaleKernel(RBFKernel(batch_size=batch_size), batch_size=batch_size)

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
        gp_model.covar_module.base_kernel.initialize(log_lengthscale=-1)
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(log_noise=0)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x1)
            loss = -mll(output, train_y1)
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for param in likelihood.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

        # Test the model
        gp_model.eval()
        likelihood.eval()

        # Update gp model to use both sine and cosine training data as train data
        gp_model.set_train_data(train_x12, train_y12, strict=False)

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
        likelihood = GaussianLikelihood()
        gp_model = ExactGPModel(train_x12, train_y12, likelihood, batch_size=2)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.covar_module.base_kernel.initialize(log_lengthscale=-1)
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(log_noise=0)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(50):
            optimizer.zero_grad()
            output = gp_model(train_x12)
            loss = -mll(output, train_y12).sum()
            loss.backward()
            optimizer.n_iter += 1
            optimizer.step()

        for param in gp_model.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for param in likelihood.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        optimizer.step()

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
