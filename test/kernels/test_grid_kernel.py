#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels import RBFKernel, GridKernel, GridInterpolationKernel
from gpytorch.lazy import KroneckerProductLazyTensor

cv = GridInterpolationKernel(RBFKernel(), grid_size=10, grid_bounds=[(0, 1), (0, 2)])
grid = cv.grid

grid_size = grid.size(-2)
grid_dim = grid.size(-1)
grid_data = torch.zeros(int(pow(grid_size, grid_dim)), grid_dim)
prev_points = None
for i in range(grid_dim):
    for j in range(grid_size):
        grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[j, i])
        if prev_points is not None:
            grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
    prev_points = grid_data[: grid_size ** (i + 1), : (i + 1)]


class TestGridKernel(unittest.TestCase):
    def test_grid_grid(self):
        base_kernel = RBFKernel()
        kernel = GridKernel(base_kernel, grid)
        grid_covar = kernel(grid_data, grid_data).evaluate_kernel()
        self.assertIsInstance(grid_covar, KroneckerProductLazyTensor)
        grid_eval = kernel(grid_data, grid_data).evaluate()
        actual_eval = base_kernel(grid_data, grid_data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_nongrid_grid(self):
        base_kernel = RBFKernel()
        data = torch.randn(5, 2)
        kernel = GridKernel(base_kernel, grid)
        grid_eval = kernel(grid_data, data).evaluate()
        actual_eval = base_kernel(grid_data, data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_nongrid_nongrid(self):
        base_kernel = RBFKernel()
        data = torch.randn(5, 2)
        kernel = GridKernel(base_kernel, grid)
        grid_eval = kernel(data, data).evaluate()
        actual_eval = base_kernel(data, data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)


if __name__ == "__main__":
    unittest.main()
