#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Normal, kl_divergence

from gpytorch.kernels import GaussianSymmetrizedKLKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestGaussianSymmetrizedKLKernel(unittest.TestCase, BaseKernelTestCase):
    def test_kernel_symkl(self):
        kernel = GaussianSymmetrizedKLKernel()
        kernel.lengthscale = 1.0

        values = torch.rand(100, 20)
        base_value = torch.zeros(1, 20)
        kernel_output = kernel(values, base_value)
        self.assertEqual(kernel_output.shape, torch.Size((100, 1)))

        value_means = values[..., :10]
        value_stds = (1e-8 + values[..., 10:].exp()) ** 0.5
        value_dist = Normal(value_means.unsqueeze(0), value_stds.unsqueeze(0))

        base_dist = Normal(torch.zeros(1, 10), torch.ones(1, 10))

        result = -(kl_divergence(value_dist, base_dist) + kl_divergence(base_dist, value_dist)).sum(-1)
        self.assertLessEqual((kernel_output.evaluate() - result.exp().transpose(-2, -1)).norm(), 1e-5)

    def create_kernel_no_ard(self, **kwargs):
        return GaussianSymmetrizedKLKernel(**kwargs)


if __name__ == "__main__":
    unittest.main()
