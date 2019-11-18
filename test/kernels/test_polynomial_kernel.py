#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import PolynomialKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PolynomialKernel(power=2, **kwargs)

    def test_computes_quadratic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=2)
        kernel.eval()

        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[i, j] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for l in range(2):
            actual[l] = kernel(a[:, l].unsqueeze(-1), b[:, l].unsqueeze(-1)).evaluate()

        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_cubic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=3)
        kernel.eval()

        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[i, j] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for l in range(2):
            actual[l] = kernel(a[:, l].unsqueeze(-1), b[:, l].unsqueeze(-1)).evaluate()

        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_quadratic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=2, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = (a[k, i].matmul(b[k, j]) + kernel.offset[k]).pow(kernel.power)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_cubic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=3, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = (a[k, i].matmul(b[k, j]) + kernel.offset[k]).pow(kernel.power)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
