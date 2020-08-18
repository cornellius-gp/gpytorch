#!/usr/bin/env python3

import math
import os
import random
import unittest

import gpytorch
import torch
from gpytorch.likelihoods import MultitaskGaussianLikelihood


# Batch training test: Let's learn hyperparameters on a sine dataset, but test on a sine dataset and a cosine dataset
# in parallel.
train_x = torch.linspace(0, 1, 20)
train_y = torch.stack([
    torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.01,
    torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.01,
    torch.sin(train_x * (2 * math.pi)) + 2 * torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.01,
    -torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.01,
], -1)


class LMCModel(gpytorch.models.ApproximateGP):
    def __init__(self):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(3, 10, 1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([3])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=4,
            num_latents=3,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3])),
            batch_shape=torch.Size([3])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
        likelihood = MultitaskGaussianLikelihood(num_tasks=4)
        model = LMCModel()

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
        # effective for VI.
        for i in range(400):
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

            for param in model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)
            for param in likelihood.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

        # Test the model
        model.eval()
        likelihood.eval()

        # Make predictions for both sets of test points, and check MAEs.
        with torch.no_grad(), gpytorch.settings.max_eager_kernel_size(1):
            batch_predictions = likelihood(model(train_x))
            preds1 = batch_predictions.mean[:, 0]
            preds2 = batch_predictions.mean[:, 1]
            preds3 = batch_predictions.mean[:, 2]
            preds4 = batch_predictions.mean[:, 3]
            mean_abs_error1 = torch.mean(torch.abs(train_y[..., 0] - preds1))
            mean_abs_error2 = torch.mean(torch.abs(train_y[..., 1] - preds2))
            mean_abs_error3 = torch.mean(torch.abs(train_y[..., 2] - preds3))
            mean_abs_error4 = torch.mean(torch.abs(train_y[..., 3] - preds4))
            self.assertLess(mean_abs_error1.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error2.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error3.squeeze().item(), 0.15)
            self.assertLess(mean_abs_error4.squeeze().item(), 0.15)

            # Smoke test for getting predictive uncertainties
            lower, upper = batch_predictions.confidence_region()
            self.assertEqual(lower.shape, train_y.shape)
            self.assertEqual(upper.shape, train_y.shape)


if __name__ == "__main__":
    unittest.main()
