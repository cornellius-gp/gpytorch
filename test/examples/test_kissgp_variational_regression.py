#!/usr/bin/env python3

from math import exp, pi

import os
import random
import torch
import unittest

import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader


# Simple training data: let's try to learn a sine function,
# but with KISS-GP let's use 100 training examples.
def make_data():
    train_x = torch.linspace(0, 1, 1000)
    train_y = torch.sin(train_x * (4 * pi)) + torch.randn(train_x.size()) * 0.2
    test_x = torch.linspace(0.02, 1, 51)
    test_y = torch.sin(test_x * (4 * pi))
    return train_x, train_y, test_x, test_y


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, grid_size=20, grid_bounds=[(-0.1, 1.1)]):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=int(pow(grid_size, len(grid_bounds)))
        )
        variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
            self, grid_size=grid_size, grid_bounds=grid_bounds, variational_distribution=variational_distribution
        )
        super(GPRegressionModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-10, 10))
        self.covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-3), exp(6), sigma=0.1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestKISSGPVariationalRegression(unittest.TestCase):
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

    def test_kissgp_gp_mean_abs_error(self):
        train_x, train_y, test_x, test_y = make_data()
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)

        model = GPRegressionModel()
        likelihood = GaussianLikelihood()
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(train_y))
        # We use SGD here, rather than Adam
        # Emperically, we find that SGD is better for variational regression
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=0.01)

        # Our loss object
        # We're using the VariationalELBO object
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        # The training loop
        def train(n_epochs=15):
            # We use a Learning rate scheduler from PyTorch to lower the learning rate during optimization
            # We're going to drop the learning rate by 1/10 after 3/4 of training
            # This helps the model converge to a minimum
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.75 * n_epochs], gamma=0.1)

            for _ in range(n_epochs):
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.float()
                    y_batch = y_batch.float()
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()

                scheduler.step()
        train()

        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for param in likelihood.parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            model.eval()
            likelihood.eval()

            test_preds = likelihood(model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        self.assertLess(mean_abs_error.squeeze().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
