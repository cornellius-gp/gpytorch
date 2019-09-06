#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels import RBFKernel, GridKernel, GridInterpolationKernel
from gpytorch.lazy import KroneckerProductLazyTensor
from gpytorch.utils import grid as grid_module

cv = GridInterpolationKernel(RBFKernel(), grid_size=10, grid_bounds=[(0, 1), (0, 2)])
grid = cv.grid

grid_size = grid[0].size(-1)
grid_dim = len(grid)
grid_data = torch.zeros(int(pow(grid_size, grid_dim)), grid_dim)
prev_points = None
for i in range(grid_dim):
    for j in range(grid_size):
        grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[i][j])
        if prev_points is not None:
            grid_data[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
    prev_points = grid_data[: grid_size ** (i + 1), : (i + 1)]


cv2 = GridInterpolationKernel(RBFKernel(), grid_size=[8, 12], grid_bounds=[(0, 1), (0, 2)])
grid2 = cv2.grid
grid_data2 = grid_module.create_data_from_grid(grid2)


class TestGridKernel(unittest.TestCase):
    def setUp(self):
        self.grid = grid
        self.grid_data = grid_data

    def test_grid_grid(self):
        base_kernel = RBFKernel()
        kernel = GridKernel(base_kernel, self.grid)
        grid_covar = kernel(self.grid_data, self.grid_data).evaluate_kernel()
        self.assertIsInstance(grid_covar, KroneckerProductLazyTensor)
        grid_eval = kernel(self.grid_data, self.grid_data).evaluate()
        actual_eval = base_kernel(self.grid_data, self.grid_data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 2e-5)

    def test_nongrid_grid(self):
        base_kernel = RBFKernel()
        data = torch.randn(5, 2)
        kernel = GridKernel(base_kernel, self.grid)
        grid_eval = kernel(self.grid_data, data).evaluate()
        actual_eval = base_kernel(self.grid_data, data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)

    def test_nongrid_nongrid(self):
        base_kernel = RBFKernel()
        data = torch.randn(5, 2)
        kernel = GridKernel(base_kernel, self.grid)
        grid_eval = kernel(data, data).evaluate()
        actual_eval = base_kernel(data, data).evaluate()
        self.assertLess(torch.norm(grid_eval - actual_eval), 1e-5)


class TestGridKernelDifferentSizes(TestGridKernel):
    def setUp(self):
        self.grid = grid2
        self.grid_data = grid_data2


class TestCreateGridFromData(unittest.TestCase):
    def test_create_grid_from_data(self):
        from itertools import product
        small_grid = [torch.tensor([1.44, 3.31]),
                      torch.tensor([-1., 2., 5.]),
                      torch.tensor([-7, 7])]
        grid_data = grid_module.create_data_from_grid(small_grid)
        order = torch.argsort(grid_data.sum(dim=1))
        grid_data = grid_data[order, :]

        # brute force creation here to make sure the efficient code is correct
        actual = torch.zeros(2*3*2, 3, dtype=torch.float)
        for r, (i, j, k) in enumerate(product(*small_grid)):
            actual[r, :] = torch.tensor([i, j, k], dtype=torch.float)
        order = torch.argsort(actual.sum(dim=1))
        actual = actual[order, :]

        self.assertLess(torch.norm(grid_data - actual), 1e-5)


class TestGridInterpolationKernel(unittest.TestCase):
    def setUp(self):
        self.test_pts1 = torch.tensor([[.9651, .6965],
                                  [.5340, .4923],
                                  [.5934, .8918],
                                  [.4249, .1549]], dtype=torch.float)
        self.test_pts2 = torch.tensor([[.5882, .6965],
                                    [.3451, .2818],
                                  [.9133, .9649],
                                       [.3561, .6711]], dtype=torch.float)

    def test_interpolation_gets_close(self):
        ski_kernel = GridInterpolationKernel(RBFKernel(), [100, 110], grid_bounds=[[0, 1], [0, 1]])
        # ski_kernel = GridInterpolationKernel(RBFKernel(), [1000, 1010], num_dims=2) # seems to be off due to boundaries?
        ski_eval = ski_kernel(self.test_pts1, self.test_pts2).evaluate()

        rbf_eval = RBFKernel()(self.test_pts1, self.test_pts2).evaluate()  # will be somewhat off due to interpolation

        self.assertLess(torch.norm(ski_eval - rbf_eval), 1e-3)

    def test_last_dim_is_batch(self):
        ski_kernel = GridInterpolationKernel(RBFKernel(), [100, 110], grid_bounds=[[0, 1], [0, 1]])
        comb_points = torch.stack([self.test_pts1, self.test_pts2], dim=2)
        ski_eval = ski_kernel(comb_points, last_dim_is_batch=True).evaluate()

        rbf_eval = RBFKernel()(comb_points, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(ski_eval - rbf_eval), 1e-3)




if __name__ == "__main__":
    unittest.main()
