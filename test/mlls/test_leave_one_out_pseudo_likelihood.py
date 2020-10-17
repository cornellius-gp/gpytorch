#!/usr/bin/env python3

import math
import unittest

import torch

import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=train_x.shape[:-2])
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class TestLeaveOneOutPseudoLikelihood(unittest.TestCase):
    def get_data(self, shapes, dtype=None, device=None):
        train_x = torch.rand(*shapes, dtype=dtype, device=device, requires_grad=True)
        train_y = torch.sin(train_x[..., 0]) + torch.cos(train_x[..., 1])
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(dtype=dtype, device=device)
        model = ExactGPModel(train_x, train_y, likelihood).to(dtype=dtype, device=device)
        loocv = gpytorch.mlls.LeaveOneOutPseudoLikelihood(likelihood=likelihood, model=model)
        return train_x, train_y, loocv

    def test_smoke(self):
        """Make sure the loocv works without batching."""
        train_x, train_y, loocv = self.get_data([5, 2])
        output = loocv.model(train_x)
        loss = -loocv(output, train_y)
        loss.backward()
        self.assertTrue(train_x.grad is not None)

    def test_smoke_batch(self):
        """Make sure the loocv works without batching."""
        train_x, train_y, loocv = self.get_data([3, 3, 3, 5, 2])
        output = loocv.model(train_x)
        loss = -loocv(output, train_y)
        assert loss.shape == (3, 3, 3)
        loss.sum().backward()
        self.assertTrue(train_x.grad is not None)

    def test_check_bordered_system(self):
        """Make sure that the bordered system solves match the naive solution."""
        n = 5
        # Compute the pseudo-likelihood via the bordered systems in O(n^3)
        train_x, train_y, loocv = self.get_data([n, 2], dtype=torch.float64)
        output = loocv.model(train_x)
        loocv_1 = loocv(output, train_y)

        # Compute the pseudo-likelihood by fitting n independent models O(n^4)
        loocv_2 = 0.0
        for i in range(n):
            inds = torch.cat((torch.arange(0, i), torch.arange(i + 1, n)))
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(train_x[inds, :], train_y[inds], likelihood)
            model.eval()
            with torch.no_grad():
                preds = likelihood(model(train_x[i, :].unsqueeze(0)))
                mean, var = preds.mean, preds.variance
                loocv_2 += -0.5 * var.log() - 0.5 * (train_y[i] - mean).pow(2.0) / var - 0.5 * math.log(2 * math.pi)
        loocv_2 /= n

        self.assertAlmostEqual(
            loocv_1.item(), loocv_2.item(),
        )


if __name__ == "__main__":
    unittest.main()
