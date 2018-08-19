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
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import InverseWishartPrior, SmoothedBoxPrior
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = torch.linspace(0, 1, 100)
y1_inds = torch.zeros(100).long()
y2_inds = torch.ones(100).long()
train_y1 = torch.sin(train_x * (2 * pi))
train_y2 = torch.cos(train_x * (2 * pi))

test_x = torch.linspace(0, 1, 51)
y1_inds_test = torch.zeros(51).long()
y2_inds_test = torch.ones(51).long()
test_y1 = torch.sin(test_x * (2 * pi))
test_y2 = torch.cos(test_x * (2 * pi))


class HadamardMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(HadamardMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(exp(-6), exp(6), sigma=0.1, log_transform=True)
        )
        self.task_covar_module = IndexKernel(n_tasks=2, rank=1, prior=InverseWishartPrior(nu=2, K=torch.eye(2)))

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar_xi = covar_x.mul(covar_i)
        return GaussianRandomVariable(mean_x, covar_xi)


class TestHadamardMultitaskGPRegression(unittest.TestCase):
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

    def test_multitask_gp_mean_abs_error(self):
        likelihood = GaussianLikelihood(log_noise_prior=SmoothedBoxPrior(-6, 6))
        gp_model = HadamardMultitaskGPModel(
            (torch.cat([train_x, train_x]), torch.cat([y1_inds, y2_inds])), torch.cat([train_y1, train_y2]), likelihood
        )
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        # Optimize the model
        gp_model.train()
        likelihood.eval()
        optimizer = optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=0.1)
        optimizer.n_iter = 0
        for _ in range(100):
            optimizer.zero_grad()
            output = gp_model(torch.cat([train_x, train_x]), torch.cat([y1_inds, y2_inds]))
            loss = -mll(output, torch.cat([train_y1, train_y2]))
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
        test_preds_task_1 = likelihood(gp_model(test_x, y1_inds_test)).mean()
        mean_abs_error_task_1 = torch.mean(torch.abs(test_y1 - test_preds_task_1))

        self.assertLess(mean_abs_error_task_1.item(), 0.05)

        test_preds_task_2 = likelihood(gp_model(test_x, y2_inds_test)).mean()
        mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds_task_2))

        self.assertLess(mean_abs_error_task_2.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
