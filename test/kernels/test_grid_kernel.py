#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import KroneckerProductLinearOperator

import gpytorch
from gpytorch.kernels import GridKernel, LinearKernel, RBFKernel
from gpytorch.utils.grid import create_data_from_grid

grid = [torch.linspace(0, 1, 5), torch.linspace(0, 2, 3), torch.linspace(0, 2, 4)]
d = len(grid)
grid_data = create_data_from_grid(grid)


class TestGridKernel(unittest.TestCase):
    def test_grid(self):
        base_kernel = RBFKernel(ard_num_dims=d)
        kernel = GridKernel(base_kernel, grid)
        with gpytorch.settings.lazily_evaluate_kernels(False):
            grid_covar = kernel(grid_data, grid_data)
        self.assertIsInstance(grid_covar, KroneckerProductLinearOperator)
        grid_eval = grid_covar.to_dense()
        actual_eval = base_kernel(grid_data, grid_data).to_dense()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

        grid_covar_diag = kernel(grid_data, diag=True)
        actual_diag = base_kernel(grid_data, grid_data, diag=True)
        self.assertLess(torch.norm(grid_covar_diag - actual_diag), 2e-5)

    def test_nongrid(self):
        base_kernel = RBFKernel(ard_num_dims=d)
        data = torch.randn(5, d)
        kernel = GridKernel(base_kernel, grid)
        with gpytorch.settings.lazily_evaluate_kernels(False), self.assertWarnsRegex(RuntimeWarning, "non-grid"):
            grid_eval = kernel(data, grid_data).to_dense()
        actual_eval = base_kernel(data, grid_data).to_dense()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_non_stationary_base(self):
        base_kernel = LinearKernel()
        with self.assertRaisesRegex(RuntimeError, "The base_kernel for GridKernel must be stationary."):
            GridKernel(base_kernel, grid)


if __name__ == "__main__":
    unittest.main()
