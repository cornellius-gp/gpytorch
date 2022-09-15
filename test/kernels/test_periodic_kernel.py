#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import PeriodicKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPeriodicKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PeriodicKernel(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return PeriodicKernel(ard_num_dims=num_dims, **kwargs)

    def test_computes_periodic_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2
        period = 3
        kernel = PeriodicKernel().initialize(lengthscale=lengthscale, period_length=period)
        kernel.eval()

        actual = torch.zeros(3, 2)
        for i in range(3):
            for j in range(2):
                val = 2 * torch.pow(torch.sin(math.pi * (a[i] - b[j]) / 3), 2) / lengthscale
                actual[i, j] = torch.exp(-val).item()

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_is_pd(self):
        # ensures 1d input is positive definite with additional jitter
        x = torch.randn(100).reshape(-1, 1)
        kernel = PeriodicKernel()
        with torch.no_grad():
            K = kernel(x, x).to_dense() + 1e-4 * torch.eye(len(x))
            eig = torch.linalg.eigvalsh(K)
            self.assertTrue((eig > 0.0).all().item())

    def test_multidimensional_inputs(self):
        # test taken from issue #835
        # ensures multidimensional input results in a positive definite kernel matrix, with additional jitter
        x = torch.randn(1000, 2)
        kernel = PeriodicKernel()
        with torch.no_grad():
            K = kernel(x, x).to_dense() + 1e-4 * torch.eye(len(x))
            eig = torch.linalg.eigvalsh(K)
            self.assertTrue((eig > 0.0).all().item())

    def test_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2], [-1, 2]], dtype=torch.float).view(2, 2, 1)
        period = torch.tensor(1, dtype=torch.float).view(1, 1, 1)
        lengthscale = torch.tensor(2, dtype=torch.float).view(1, 1, 1)
        kernel = PeriodicKernel().initialize(lengthscale=lengthscale, period_length=period)
        kernel.eval()

        actual = torch.zeros(2, 3, 2)
        for k in range(2):
            actual[k] = kernel(a[k], b[k]).to_dense()

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch_separate(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2], [-1, 2]], dtype=torch.float).view(2, 2, 1)
        period = torch.tensor([1, 2], dtype=torch.float).view(2, 1, 1)
        lengthscale = torch.tensor([2, 1], dtype=torch.float).view(2, 1, 1)
        kernel = PeriodicKernel(batch_shape=torch.Size([2])).initialize(lengthscale=lengthscale, period_length=period)
        kernel.eval()

        actual = torch.zeros(2, 3, 2)
        for k in range(2):
            diff = a[k].unsqueeze(1) - b[k].unsqueeze(0)
            diff = diff * math.pi / period[k].unsqueeze(-1)
            actual[k] = diff.sin().pow(2).sum(dim=-1).div(lengthscale[k]).mul(-2.0).exp_()

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def create_kernel_with_prior(self, period_length_prior):
        return PeriodicKernel(period_length_prior=period_length_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        kernel_fn = lambda prior: self.create_kernel_with_prior(prior)
        kernel_fn(None)
        kernel_fn(NormalPrior(0, 1))
        self.assertRaises(TypeError, kernel_fn, 1)


if __name__ == "__main__":
    unittest.main()
