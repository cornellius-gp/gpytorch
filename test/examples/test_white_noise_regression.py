from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import exp, pi

import os
import random
import torch
import unittest
import gpytorch
from torch import optim
from gpytorch.kernels import RBFKernel, WhiteNoiseKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.distributions import MultivariateNormal


# Simple training data: let's try to learn a sine function
train_x = torch.linspace(0, 1, 11)
train_y = torch.sin(train_x * (2 * pi))

test_x = torch.linspace(0, 1, 51)
test_y = torch.sin(test_x * (2 * pi))


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.rbf_covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1, log_transform=True)
        )
        self.noise_covar_module = WhiteNoiseKernel(variances=torch.ones(11) * 0.001)
        self.covar_module = ScaleKernel(self.rbf_covar_module + self.noise_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestSimpleGPRegression(unittest.TestCase):
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

    def test_posterior_latent_gp_and_likelihood_without_optimization(self):
        with gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to be ridiculous
            likelihood = GaussianLikelihood(
                log_noise_prior=SmoothedBoxPrior(exp(-10), exp(10), sigma=0.25, log_transform=True)
            )
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            # Update lengthscale prior to accommodate extreme parameters
            gp_model.rbf_covar_module.set_parameter_priors(
                log_lengthscale=SmoothedBoxPrior(exp(-10), exp(10), sigma=0.5, log_transform=True)
            )
            gp_model.rbf_covar_module.initialize(log_lengthscale=-10)
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(log_noise=-10)

            # Compute posterior distribution
            gp_model.eval()
            likelihood.eval()

            # Let's see how our model does, conditioned with weird hyperparams
            # The posterior should fit all the data
            function_predictions = likelihood(gp_model(train_x))

            self.assertLess(torch.norm(function_predictions.mean - train_y), 1e-3)
            self.assertLess(torch.norm(function_predictions.variance), 5e-3)

            # It shouldn't fit much else though
            test_function_predictions = gp_model(torch.tensor([1.1], dtype=torch.float))

            self.assertLess(torch.norm(test_function_predictions.mean - 0), 1e-4)
            self.assertLess(torch.norm(test_function_predictions.variance - 1), 1e-4)

    def test_posterior_latent_gp_and_likelihood_with_optimization(self):
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(
            log_noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1, log_transform=True)
        )
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.rbf_covar_module.initialize(log_lengthscale=1)
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(log_noise=1)

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        with gpytorch.settings.debug(False):
            for _ in range(50):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
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
            test_function_predictions = likelihood(gp_model(test_x))
            mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean))

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var(self):
        with gpytorch.fast_pred_var(), gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to
            # something they shouldn't be
            likelihood = GaussianLikelihood(
                log_noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1, log_transform=True)
            )
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.rbf_covar_module.initialize(log_lengthscale=1)
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(log_noise=1)

            # Find optimal model hyperparameters
            gp_model.train()
            likelihood.train()
            optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
            optimizer.n_iter = 0
            for _ in range(50):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
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
            # Set the cache
            test_function_predictions = likelihood(gp_model(train_x))

            # Now bump up the likelihood to something huge
            # This will make it easy to calculate the variance
            likelihood.log_noise.data.fill_(3)
            test_function_predictions = likelihood(gp_model(train_x))

            noise = likelihood.log_noise.exp()
            var_diff = (test_function_predictions.variance - noise).abs()

            self.assertLess(torch.max(var_diff / noise), 0.05)

    def test_posterior_latent_gp_and_likelihood_with_optimization_cuda(self):
        if torch.cuda.is_available():
            # We're manually going to set the hyperparameters to
            # something they shouldn't be
            likelihood = GaussianLikelihood(
                log_noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1, log_transform=True)
            ).cuda()
            gp_model = ExactGPModel(train_x.cuda(), train_y.cuda(), likelihood).cuda()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.rbf_covar_module.initialize(log_lengthscale=1)
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(log_noise=1)

            # Find optimal model hyperparameters
            gp_model.train()
            likelihood.train()
            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
            optimizer.n_iter = 0
            with gpytorch.settings.debug(False):
                for _ in range(50):
                    optimizer.zero_grad()
                    output = gp_model(train_x.cuda())
                    loss = -mll(output, train_y.cuda())
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
                test_function_predictions = likelihood(gp_model(test_x.cuda()))
                mean_abs_error = torch.mean(torch.abs(test_y.cuda() - test_function_predictions.mean))

            self.assertLess(mean_abs_error.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
