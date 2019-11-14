#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import LinearKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestLinearKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return LinearKernel(**kwargs)

    def test_computes_linear_function_rectangular(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 1], dtype=torch.float).view(3, 1)

        kernel = LinearKernel().initialize(variance=1.0)
        kernel.eval()
        actual = torch.matmul(a, b.t())
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_computes_linear_function_square(self):
        a = torch.tensor([[4, 1], [2, 0], [8, 3]], dtype=torch.float)

        kernel = LinearKernel().initialize(variance=3.14)
        kernel.eval()
        actual = torch.matmul(a, a.t()) * 3.14
        res = kernel(a, a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, a).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims
        dim_group_a = a
        dim_group_a = dim_group_a.permute(1, 0).reshape(-1, 3)
        actual = 3.14 * torch.mul(dim_group_a.unsqueeze(-1), dim_group_a.unsqueeze(-2))
        res = kernel(a, a, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims + diag
        res = kernel(a, a, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_computes_linear_function_square_batch(self):
        a = torch.tensor([[[4, 1], [2, 0], [8, 3]], [[1, 1], [2, 1], [1, 3]]], dtype=torch.float)

        kernel = LinearKernel().initialize(variance=1.0)
        kernel.eval()
        actual = torch.matmul(a, a.transpose(-1, -2))
        res = kernel(a, a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, a).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims
        dim_group_a = a
        dim_group_a = dim_group_a.transpose(-1, -2).unsqueeze(-1)
        actual = dim_group_a.matmul(dim_group_a.transpose(-2, -1))
        res = kernel(a, a, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims + diag
        res = kernel(a, a, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-4)


if __name__ == "__main__":
    unittest.main()
