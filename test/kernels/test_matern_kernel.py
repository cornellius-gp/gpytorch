#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import MaternKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestMatern25BaseKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return MaternKernel(nu=2.5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return MaternKernel(nu=2.5, ard_num_dims=num_dims, **kwargs)


class TestMatern05BaseKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        kernel = MaternKernel(nu=0.5, **kwargs)
        kernel.initialize(lengthscale=5.0)
        return kernel

    def create_kernel_ard(self, num_dims, **kwargs):
        kernel = MaternKernel(nu=0.5, ard_num_dims=num_dims, **kwargs)
        kernel.initialize(lengthscale=5.0)
        return kernel


class TestMaternKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return MaternKernel(nu=1.5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return MaternKernel(nu=1.5, ard_num_dims=num_dims, **kwargs)

    def test_forward_nu_1_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=0.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        actual = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).div_(-lengthscale).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_3_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=1.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(3) / lengthscale)
        actual = (dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_5_over_2(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=2.5).initialize(lengthscale=lengthscale)
        kernel.eval()

        dist = torch.tensor([[4, 2], [2, 0], [8, 6]], dtype=torch.float).mul_(math.sqrt(5) / lengthscale)
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
        b = torch.tensor([[1, 4], [1, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = MaternKernel(nu=2.5, ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[1, 1], [2, 2]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor([[[0, 0], [2, 2]], [[1, 1], [0, 0]]], dtype=torch.float)
        dist.mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[[1], [1]], [[2], [0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 3]], [[2, -1, 2], [2, -1, 0]]], dtype=torch.float)
        b = torch.tensor([[[1, 4, 3]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = MaternKernel(nu=2.5, batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.eval()

        dist = torch.tensor([[[1, 1], [1, 1]], [[4, 4], [0, 0]]], dtype=torch.float).mul_(math.sqrt(5))
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        dist = torch.tensor(
            [
                [[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]], [[4.0, 4.0], [0.0, 0.0]]],
            ]
        )

        dist.mul_(math.sqrt(5))
        dist = dist.view(3, 2, 2, 2).transpose(0, 1)
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
