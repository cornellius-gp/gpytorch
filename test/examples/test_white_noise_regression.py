#!/usr/bin/env python3

import os
import random
import unittest
from math import exp, pi

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel, WhiteNoiseKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from torch import optim


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.rbf_covar_module = RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
        self.noise_covar_module = WhiteNoiseKernel(variances=torch.ones(11) * 0.001)
        self.covar_module = ScaleKernel(self.rbf_covar_module + self.noise_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestWhiteNoiseGPRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def _get_data(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # Simple training data: let's try to learn a sine function
        train_x = torch.linspace(0, 1, 11, device=device)
        train_y = torch.sin(train_x * (2 * pi))
        test_x = torch.linspace(0, 1, 51, device=device)
        test_y = torch.sin(test_x * (2 * pi))
        return train_x, test_x, train_y, test_y

    def test_posterior_latent_gp_and_likelihood_without_optimization(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        with gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to be ridiculous
            likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-10), exp(10), sigma=0.25))
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            # Update lengthscale prior to accommodate extreme parameters
            gp_model.rbf_covar_module.register_prior(
                "log_lengthscale_prior", SmoothedBoxPrior(exp(-10), exp(10), sigma=0.5), "raw_lengthscale"
            )
            gp_model.rbf_covar_module.initialize(log_lengthscale=-10)
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(log_noise=-10)

            if cuda:
                gp_model.cuda()
                likelihood.cuda()

            # Compute posterior distribution
            gp_model.eval()
            likelihood.eval()

            # Let's see how our model does, conditioned with weird hyperparams
            # The posterior should fit all the data
            function_predictions = likelihood(gp_model(train_x))

            self.assertLess(torch.norm(function_predictions.mean - train_y), 1e-3)
            self.assertLess(torch.norm(function_predictions.variance), 5e-3)

            # It shouldn't fit much else though
            test_function_predictions = gp_model(torch.tensor([1.1]).type_as(test_x))

            self.assertLess(torch.norm(test_function_predictions.mean - 0), 1e-4)
            self.assertLess(torch.norm(test_function_predictions.variance - gp_model.covar_module.outputscale), 1e-4)

    def test_posterior_latent_gp_and_likelihood_without_optimization_cuda(self):
        if torch.cuda.is_available():
            self.test_posterior_latent_gp_and_likelihood_without_optimization(cuda=True)

    def test_posterior_latent_gp_and_likelihood_with_optimization(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.rbf_covar_module.initialize(log_lengthscale=1)
        gp_model.mean_module.initialize(constant=0)
        likelihood.initialize(log_noise=1)

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        with gpytorch.settings.debug(False):
            for _ in range(75):
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

    def test_posterior_latent_gp_and_likelihood_with_optimization_cuda(self):
        if torch.cuda.is_available():
            self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=True)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to something they shouldn't be
            likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.rbf_covar_module.initialize(log_lengthscale=1)
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(log_noise=1)

            if cuda:
                gp_model.cuda()
                likelihood.cuda()

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
            likelihood.raw_noise.data.fill_(3)
            test_function_predictions = likelihood(gp_model(train_x))

            noise = likelihood.noise
            var_diff = (test_function_predictions.variance - noise).abs()

            self.assertLess(torch.max(var_diff / noise), 0.05)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var_cuda(self):
        if torch.cuda.is_available():
            self.test_posterior_latent_gp_and_likelihood_fast_pred_var(cuda=True)


if __name__ == "__main__":
    unittest.main()
