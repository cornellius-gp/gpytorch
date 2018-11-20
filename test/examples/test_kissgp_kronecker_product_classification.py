#!/usr/bin/env python3

from math import exp

import os
import random
import torch
import unittest
import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal

n = 4
train_x = torch.zeros(pow(n, 2), 2)
train_y = torch.zeros(pow(n, 2))
for i in range(n):
    for j in range(n):
        train_x[i * n + j][0] = float(i) / (n - 1)
        train_x[i * n + j][1] = float(j) / (n - 1)
        train_y[i * n + j] = pow(-1, int(i / 2) + int(j / 2))
train_x = train_x
train_y = train_y


class GPClassificationModel(gpytorch.models.GridInducingVariationalGP):
    def __init__(self):
        super(GPClassificationModel, self).__init__(grid_size=8, grid_bounds=[(0, 3), (0, 3)])
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=2, lengthscale_prior=SmoothedBoxPrior(exp(-2.5), exp(3), sigma=0.1))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred


class TestKISSGPKroneckerProductClassification(unittest.TestCase):
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
        model = GPClassificationModel()
        likelihood = BernoulliLikelihood()
        mll = gpytorch.mlls.VariationalMarginalLogLikelihood(likelihood, model, num_data=len(train_y))

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        with gpytorch.settings.max_preconditioner_size(5):
            optimizer = optim.Adam(model.parameters(), lr=0.15)
            optimizer.n_iter = 0
            for _ in range(100):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.n_iter += 1
                optimizer.step()

            for param in model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Set back to eval mode
            model.eval()
            likelihood.eval()

            test_preds = model(train_x).mean.ge(0.5).float().mul(2).sub(1).squeeze()
            mean_abs_error = torch.mean(torch.abs(train_y - test_preds) / 2)
            self.assertLess(mean_abs_error.squeeze().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
