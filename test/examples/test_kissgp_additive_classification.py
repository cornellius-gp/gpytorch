#!/usr/bin/env python3

from math import exp

import os
import random
import torch
import unittest

import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.variational import AdditiveGridInterpolationVariationalStrategy, CholeskyVariationalDistribution

n = 64
train_x = torch.zeros(n ** 2, 2)
train_x[:, 0].copy_(torch.linspace(-1, 1, n).repeat(n))
train_x[:, 1].copy_(torch.linspace(-1, 1, n).unsqueeze(1).repeat(1, n).view(-1))
train_y = train_x[:, 0].abs().lt(0.5).float()
train_y = train_y * (train_x[:, 1].abs().lt(0.5)).float()
train_y = train_y.float()


class GPClassificationModel(ApproximateGP):
    def __init__(self, grid_size=16, grid_bounds=([-1, 1],)):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=16, batch_shape=torch.Size([2]))
        variational_strategy = AdditiveGridInterpolationVariationalStrategy(
            self,
            grid_size=grid_size,
            grid_bounds=grid_bounds,
            num_dim=2,
            variational_distribution=variational_distribution,
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=1, lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)),
            outputscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestKISSGPAdditiveClassification(unittest.TestCase):
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

    def test_kissgp_classification_error(self):
        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(5):
            model = GPClassificationModel()
            likelihood = BernoulliLikelihood()
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            optimizer = optim.Adam(model.parameters(), lr=0.15)
            optimizer.n_iter = 0
            for _ in range(25):
                optimizer.zero_grad()
                # Get predictive output
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y).sum()
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Set back to eval mode
            model.eval()
            likelihood.eval()

            test_preds = model(train_x).mean.ge(0.5).float()
            mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)

        self.assertLess(mean_abs_error.squeeze().item(), 0.15)


if __name__ == "__main__":
    unittest.main()
