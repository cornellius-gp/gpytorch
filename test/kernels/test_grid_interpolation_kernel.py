#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import GridInterpolationKernel, RBFKernel
from gpytorch.lazy import InterpolatedLazyTensor


class TestGridInterpolationKernel(unittest.TestCase):
    def test_standard(self):
        base_kernel = RBFKernel(ard_num_dims=2)
        kernel = GridInterpolationKernel(base_kernel, num_dims=2, grid_size=128, grid_bounds=[(-1.2, 1.2)] * 2)

        xs = torch.randn(5, 2).clamp(-1, 1)
        interp_covar = kernel(xs, xs).evaluate_kernel()
        self.assertIsInstance(interp_covar, InterpolatedLazyTensor)

        xs = torch.randn(5, 2).clamp(-1, 1)
        grid_eval = kernel(xs, xs).evaluate()
        actual_eval = base_kernel(xs, xs).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

        xs = torch.randn(3, 5, 2).clamp(-1, 1)
        grid_eval = kernel(xs, xs).evaluate()
        actual_eval = base_kernel(xs, xs).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

    def test_batch_base_kernel(self):
        base_kernel = RBFKernel(batch_shape=torch.Size([3]), ard_num_dims=2)
        kernel = GridInterpolationKernel(base_kernel, num_dims=2, grid_size=128, grid_bounds=[(-1.2, 1.2)] * 2)

        xs = torch.randn(5, 2).clamp(-1, 1)
        grid_eval = kernel(xs, xs).evaluate()
        actual_eval = base_kernel(xs, xs).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

        xs = torch.randn(3, 5, 2).clamp(-1, 1)
        grid_eval = kernel(xs, xs).evaluate()
        actual_eval = base_kernel(xs, xs).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

        xs = torch.randn(4, 3, 5, 2).clamp(-1, 1)
        grid_eval = kernel(xs, xs).evaluate()
        actual_eval = base_kernel(xs, xs).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)


if __name__ == "__main__":
    unittest.main()
