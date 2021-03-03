#!/usr/bin/env python3

import os
import random
import unittest
import warnings
from math import exp, pi

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.warnings import GPInputWarning
from torch import optim


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.rbf_covar_module = RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
        self.covar_module = ScaleKernel(self.rbf_covar_module)

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

    def test_posterior_latent_gp_and_likelihood_with_optimization(self, cuda=False, cg=False):
        # This test throws a warning because the fixed noise likelihood gets the wrong input
        warnings.simplefilter("ignore", GPInputWarning)

        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        # We're manually going to set the hyperparameters to something they shouldn't be
        likelihood = FixedNoiseGaussianLikelihood(torch.ones(11) * 0.001)
        gp_model = ExactGPModel(train_x, train_y, likelihood)
        mll = gpytorch.ExactMarginalLogLikelihood(likelihood, gp_model)
        gp_model.rbf_covar_module.initialize(lengthscale=exp(1))
        gp_model.mean_module.initialize(constant=0)

        if cuda:
            gp_model.cuda()
            likelihood.cuda()

        # Find optimal model hyperparameters
        gp_model.train()
        likelihood.train()

        optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
        optimizer.n_iter = 0
        with gpytorch.settings.debug(False), gpytorch.settings.max_cholesky_size(0 if cg else 1e10):
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
            optimizer.step()

            # Test the model
            gp_model.eval()
            likelihood.eval()
            test_function_predictions = likelihood(gp_model(test_x))
            mean_abs_error = torch.mean(torch.abs(test_y - test_function_predictions.mean))

        self.assertLess(mean_abs_error.squeeze().item(), 0.05)

    def test_posterior_latent_gp_and_likelihood_with_optimization_with_cg(self):
        return self.test_posterior_latent_gp_and_likelihood_with_optimization(cg=True)

    def test_posterior_latent_gp_and_likelihood_with_optimization_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_posterior_latent_gp_and_likelihood_with_optimization(cuda=True)

    def test_posterior_latent_gp_and_likelihood_fast_pred_var(self, cuda=False):
        train_x, test_x, train_y, test_y = self._get_data(cuda=cuda)
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.debug(False):
            # We're manually going to set the hyperparameters to something they shouldn't be
            likelihood = GaussianLikelihood(noise_prior=SmoothedBoxPrior(exp(-3), exp(3), sigma=0.1))
            gp_model = ExactGPModel(train_x, train_y, likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)
            gp_model.rbf_covar_module.initialize(lengthscale=exp(1))
            gp_model.mean_module.initialize(constant=0)
            likelihood.initialize(noise=exp(1))

            if cuda:
                gp_model.cuda()
                likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            likelihood.train()
            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
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
            with least_used_cuda_device():
                self.test_posterior_latent_gp_and_likelihood_fast_pred_var(cuda=True)


if __name__ == "__main__":
    unittest.main()
