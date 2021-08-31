#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import PiecewisePolynomialKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPiecewisePolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PiecewisePolynomialKernel(q=2, **kwargs)

    def test_computes_piecewise_polynomial_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PiecewisePolynomialKernel(q=0)
        kernel.eval()

        def test_r(a, b):
            return torch.cdist(a, b)

        def test_get_cov(r, j, q):
            if q == 0:
                return 1
            if q == 1:
                return (j + 1) * r + 1
            if q == 2:
                return 1 + (j + 2) * r + ((j ** 2 + 4 * j + 3) / 3.0) * r ** 2
            if q == 3:
                return (
                    1
                    + (j + 3) * r
                    + ((6 * j ** 2 + 36 * j + 45) / 15.0) * r ** 2
                    + ((j ** 3 + 9 * j ** 2 + 23 * j + 15) / 15.0) * r ** 3
                )

        def test_fmax(r, j, q):
            return torch.max(torch.tensor(0.0), 1 - r).pow(j + q)

        actual = torch.zeros(3, 3)
        j = torch.floor(a / 2.0).shape[-1] + kernel.q + 1
        r = test_r(a, b)
        actual = test_fmax(r, j, kernel.q) * test_get_cov(r, j, kernel.q)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        actual = actual.diag()
        res = kernel(a, b).diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[:, i].unsqueeze(-1), b[:, i].unsqueeze(-1)).evaluate()

        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_piecewise_polynomial_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PiecewisePolynomialKernel(q=0, batch_shape=torch.Size([2]))
        kernel.eval()

        def test_r(a, b):
            return torch.cdist(a, b)

        def test_get_cov(r, j, q):
            if q == 0:
                return 1
            if q == 1:
                return (j + 1) * r + 1
            if q == 2:
                return 1 + (j + 2) * r + ((j ** 2 + 4 * j + 3) / 3.0) * r ** 2
            if q == 3:
                return (
                    1
                    + (j + 3) * r
                    + ((6 * j ** 2 + 36 * j + 45) / 15.0) * r ** 2
                    + ((j ** 3 + 9 * j ** 2 + 23 * j + 15) / 15.0) * r ** 3
                )

        def test_fmax(r, j, q):
            return torch.max(torch.tensor(0.0), 1 - r).pow(j + q)

        actual = torch.zeros(3, 3)
        j = torch.floor(a / 2.0).shape[-1] + kernel.q + 1
        r = test_r(a, b)
        actual = test_fmax(r, j, kernel.q) * test_get_cov(r, j, kernel.q)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
