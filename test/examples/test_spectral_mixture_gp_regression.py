from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import exp, pi

import os
import torch
import unittest
import gpytorch
from torch import optim
from torch.autograd import Variable
from gpytorch.kernels import SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 0.75, 11))
train_y = Variable(torch.sin(train_x.data * (2 * pi)))

# Spectral mixture kernel should be able to train on
# data up to x=0.75, but test on data up to x=2
test_x = Variable(torch.linspace(0, 2, 51))
test_y = Variable(torch.sin(test_x.data * (2 * pi)))


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = SpectralMixtureKernel(
            n_mixtures=3,
            log_mixture_weight_prior=SmoothedBoxPrior(
                torch.ones(3).fill_(-5).exp_(), torch.ones(3).fill_(-5).exp_(), sigma=0.1, log_transform=True
            ),
            log_mixture_mean_prior=SmoothedBoxPrior(
                torch.ones(3).fill_(-5).exp_(), torch.ones(3).fill_(-5).exp_(), sigma=0.1, log_transform=True
            ),
            log_mixture_scale_prior=SmoothedBoxPrior(
                torch.ones(3).fill_(-5).exp_(), torch.ones(3).fill_(-5).exp_(), sigma=0.1, log_transform=True
            ),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return GaussianRandomVariable(mean_x, covar_x)


class TestSpectralMixtureGPRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_spectral_mixture_gp_mean_abs_error(self):
        likelihood = GaussianLikelihood(
            log_noise_prior=SmoothedBoxPrior(exp(-5), exp(5), sigma=0.1, log_transform=True)
        )
        gp_model = SpectralMixtureGPModel(train_x.data, train_y.data, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.train()
        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0

        with gpytorch.settings.num_trace_samples(100):
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
            test_preds = likelihood(gp_model(test_x)).mean()
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))

        # The spectral mixture kernel should be trivially able to
        # extrapolate the sine function.
        self.assertLess(mean_abs_error.data.squeeze().item(), 0.15)


if __name__ == "__main__":
    unittest.main()
