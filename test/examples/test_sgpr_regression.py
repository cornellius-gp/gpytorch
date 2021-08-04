#!/usr/bin/env python3

import os
import random
import unittest
from unittest.mock import MagicMock, patch
import warnings
from math import exp, pi

import gpytorch
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.test.utils import least_used_cuda_device
from gpytorch.utils.cholesky import CHOLESKY_METHOD
from gpytorch.utils.warnings import NumericalWarning
from torch import optim

from gpytorch.test.base_test_case import BaseTestCase


# Simple training data: let's try to learn a sine function,
# but with SGPR
# let's use 100 training examples.
def make_data(cuda=False):
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2 * pi))
    train_y.add_(torch.randn_like(train_y), alpha=1e-2)
    test_x = torch.rand(51)
    test_y = torch.sin(test_x * (2 * pi))
    if cuda:
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
    return train_x, train_y, test_x, test_y


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(prior=SmoothedBoxPrior(-1e-5, 1e-5))
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale_prior=SmoothedBoxPrior(exp(-5), exp(6), sigma=0.1)))
        self.covar_module = InducingPointKernel(
            self.base_covar_module, inducing_points=torch.linspace(0, 1, 32), likelihood=likelihood
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class TestSGPRRegression(unittest.TestCase, BaseTestCase):
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

    def test_sgpr_mean_abs_error(self, cuda=False):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        train_x, train_y, test_x, test_y = make_data(cuda=cuda)
        likelihood = GaussianLikelihood()
        gp_model = GPRegressionModel(train_x, train_y, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

        if cuda:
            gp_model = gp_model.cuda()
            likelihood = likelihood.cuda()

        # Mock cholesky
        _wrapped_cholesky = MagicMock(
            wraps=torch.linalg.cholesky if CHOLESKY_METHOD == "torch.linalg.cholesky" else torch.linalg.cholesky_ex
        )
        with patch(CHOLESKY_METHOD, new=_wrapped_cholesky) as cholesky_mock:

            # Optimize the model
            gp_model.train()
            likelihood.train()

            optimizer = optim.Adam(gp_model.parameters(), lr=0.1)
            for _ in range(30):
                optimizer.zero_grad()
                output = gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

                # Check that we have the right LazyTensor type
                kernel = likelihood(gp_model(train_x)).lazy_covariance_matrix.evaluate_kernel()
                self.assertIsInstance(kernel, gpytorch.lazy.LowRankRootAddedDiagLazyTensor)

            for param in gp_model.parameters():
                self.assertTrue(param.grad is not None)
                self.assertGreater(param.grad.norm().item(), 0)

            # Test the model
            gp_model.eval()
            likelihood.eval()

            test_preds = likelihood(gp_model(test_x)).mean
            mean_abs_error = torch.mean(torch.abs(test_y - test_preds))
            cholesky_mock.assert_called()  # We SHOULD call Cholesky...
            for chol_arg in cholesky_mock.call_args_list:
                first_arg = chol_arg[0][0]
                self.assertTrue(torch.is_tensor(first_arg))
                self.assertTrue(first_arg.size(-1) == gp_model.covar_module.inducing_points.size(-2))

        self.assertLess(mean_abs_error.squeeze().item(), 0.1)

        # Test variances
        test_vars = likelihood(gp_model(test_x)).variance
        self.assertAllClose(test_vars, likelihood(gp_model(test_x)).covariance_matrix.diagonal(dim1=-1, dim2=-2))
        self.assertGreater(test_vars.min().item() + 0.1, likelihood.noise.item())
        self.assertLess(
            test_vars.max().item() - 0.05,
            likelihood.noise.item() + gp_model.covar_module.base_kernel.outputscale.item()
        )

        # Test on training data
        test_outputs = likelihood(gp_model(train_x))
        self.assertLess((test_outputs.mean - train_y).max().item(), 0.1)
        self.assertLess(test_outputs.variance.max().item(), likelihood.noise.item() * 2)

    def test_sgpr_mean_abs_error_cuda(self):
        # Suppress numerical warnings
        warnings.simplefilter("ignore", NumericalWarning)

        if not torch.cuda.is_available():
            return

        with least_used_cuda_device():
            self.test_sgpr_mean_abs_error(cuda=True)


if __name__ == "__main__":
    unittest.main()
