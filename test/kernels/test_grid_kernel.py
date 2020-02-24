#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.kernels import GridKernel, LinearKernel, RBFKernel
from gpytorch.lazy import KroneckerProductLazyTensor
from gpytorch.utils.grid import create_data_from_grid

grid = [torch.linspace(0, 1, 5), torch.linspace(0, 2, 3)]
d = len(grid)
grid_data = create_data_from_grid(grid)


class TestGridKernel(unittest.TestCase):
    def test_grid_grid(self, toeplitz=True):
        with gpytorch.settings.use_toeplitz(toeplitz):
            base_kernel = RBFKernel(ard_num_dims=2)
            kernel = GridKernel(base_kernel, grid)
            grid_covar = kernel(grid_data, grid_data).evaluate_kernel()
            self.assertIsInstance(grid_covar, KroneckerProductLazyTensor)
            grid_eval = kernel(grid_data, grid_data).evaluate()
            actual_eval = base_kernel(grid_data, grid_data).evaluate()
            self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

    def test_grid_grid_nontoeplitz(self):
        return self.test_grid_grid(toeplitz=False)

    def test_nongrid_grid(self, toeplitz=True):
        with gpytorch.settings.use_toeplitz(toeplitz):
            base_kernel = RBFKernel(ard_num_dims=2)
            data = torch.randn(5, d)
            kernel = GridKernel(base_kernel, grid)
            grid_eval = kernel(grid_data, data).evaluate()
            actual_eval = base_kernel(grid_data, data).evaluate()
            self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_nongrid_grid_nontoeplitz(self):
        return self.test_nongrid_grid(toeplitz=False)

    def test_nongrid_nongrid(self, toeplitz=True):
        with gpytorch.settings.use_toeplitz(toeplitz):
            base_kernel = RBFKernel(ard_num_dims=2)
            data = torch.randn(5, d)
            kernel = GridKernel(base_kernel, grid)
            grid_eval = kernel(data, data).evaluate()
            actual_eval = base_kernel(data, data).evaluate()
            self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_nongrid_nongrid_nontoeplitz(self):
        return self.test_nongrid_nongrid(toeplitz=False)

    def test_non_stationary_base(self):
        base_kernel = LinearKernel()
        with self.assertRaisesRegex(RuntimeError, "The base_kernel for GridKernel must be stationary."):
            GridKernel(base_kernel, grid)


if __name__ == "__main__":
    unittest.main()
