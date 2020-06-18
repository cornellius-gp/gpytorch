#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Normal, kl_divergence

from gpytorch.kernels import DistributionalInputKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestDistributionalInputKernel(unittest.TestCase, BaseKernelTestCase):
    def test_kernel_symkl(self):
        kernel = DistributionalInputKernel(distance="symmetrized_kl", num_dims=10)
        kernel.lengthscale = 1.0

        values = torch.rand(100, 20)
        base_value = torch.zeros(1, 20)
        kernel_output = kernel(values, base_value)
        self.assertEqual(kernel_output.shape, torch.size(100, 1))

        value_means = values[..., :10]
        value_stds = (1e-8 + values[..., 10:].exp()) ** 0.5
        value_dist = Normal(value_means, value_stds)

        base_dist = Normal(torch.zeros(1), torch.ones(1))

        result = -(kl_divergence(value_dist, base_dist) + kl_divergence(base_dist, value_dist))
        self.assertLessEqual((kernel_output - result.exp()).norm(), 1e-5)

    def test_batched_kernel_symkl(self):
        kernel = DistributionalInputKernel(distance="symmetrized_kl", num_dims=10)
        kernel.lengthscale = 1.0

        values = torch.rand(3, 100, 20)
        base_value = torch.zeros(3, 1, 20)
        kernel_output = kernel(values, base_value)
        self.assertEqual(kernel_output.shape, torch.size(3, 100, 1))

        value_means = values[..., :10]
        value_stds = (1e-8 + values[..., 10:].exp()) ** 0.5
        value_dist = Normal(value_means, value_stds)

        base_dist = Normal(torch.zeros(3, 1), torch.ones(3, 1))

        result = -(kl_divergence(value_dist, base_dist) + kl_divergence(base_dist, value_dist))
        self.assertLessEqual((kernel_output - result.exp()).norm(), 1e-5)


if __name__ == "__main__":
    unittest.main()
