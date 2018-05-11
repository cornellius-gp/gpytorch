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
from gpytorch.kernels import RBFKernel, IndexKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.random_variables import GaussianRandomVariable

# Simple training data: let's try to learn a sine function
train_x = Variable(torch.linspace(0, 1, 11))
y1_inds = Variable(torch.zeros(11).long())
y2_inds = Variable(torch.ones(11).long())
train_y1 = Variable(torch.sin(train_x.data * (2 * pi)))
train_y2 = Variable(torch.cos(train_x.data * (2 * pi)))

test_x = Variable(torch.linspace(0, 1, 51))
y1_inds_test = Variable(torch.zeros(51).long())
y2_inds_test = Variable(torch.ones(51).long())
test_y1 = Variable(torch.sin(test_x.data * (2 * pi)))
test_y2 = Variable(torch.cos(test_x.data * (2 * pi)))


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1, 1))
        self.covar_module = RBFKernel(
            log_lengthscale_prior=SmoothedBoxPrior(exp(-6), exp(6), sigma=0.1, log_transform=True)
        )
        self.task_covar_module = IndexKernel(
            n_tasks=2,
            rank=1,
            covar_factor_prior=SmoothedBoxPrior(
                torch.ones(2).fill_(-6).exp_(), torch.ones(2).fill_(6).exp_(), sigma=0.1, log_transform=True
            ),
            log_var_prior=SmoothedBoxPrior(
                torch.ones(2).fill_(-6).exp_(), torch.ones(2).fill_(6).exp_(), sigma=0.1, log_transform=True
            ),
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(i)
        covar_xi = covar_x.mul(covar_i)
        return GaussianRandomVariable(mean_x, covar_xi)


class TestMultiTaskGPRegression(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_multitask_gp_mean_abs_error(self):
        likelihood = GaussianLikelihood(log_noise_prior=SmoothedBoxPrior(-6, 6))
        gp_model = MultitaskGPModel(
            (torch.cat([train_x.data, train_x.data]), torch.cat([y1_inds.data, y2_inds.data])),
            torch.cat([train_y1.data, train_y2.data]),
            likelihood,
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

        self.assertLess(mean_abs_error_task_1.data.squeeze().item(), 0.05)

        test_preds_task_2 = likelihood(gp_model(test_x, y2_inds_test)).mean()
        mean_abs_error_task_2 = torch.mean(torch.abs(test_y2 - test_preds_task_2))

        self.assertLess(mean_abs_error_task_2.data.squeeze().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
