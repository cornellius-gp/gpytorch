#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import CosineKernel


class TestCosineKernel(unittest.TestCase):
    def test_computes_periodic_function(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        period = 1
        kernel = CosineKernel().initialize(period_length=period)
        kernel.eval()

        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[i, j] = torch.cos(math.pi * ((a[i] - b[j]) / period).norm(2, dim=-1))

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for i in range(3):
            for j in range(3):
                for l in range(2):
                    actual[l, i, j] = torch.cos(math.pi * ((a[i, l] - b[j, l]) / period))
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        period = torch.tensor(1, dtype=torch.float).view(1, 1, 1)
        kernel = CosineKernel().initialize(period_length=period)
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = torch.cos(math.pi * ((a[k, i] - b[k, j]) / period))

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch_separate(self):
        a = torch.tensor([[[4, 1], [2, 2], [8, 0]], [[2, 5], [6, 1], [0, 1]]], dtype=torch.float)
        b = torch.tensor([[[0, 0], [2, 1], [1, 0]], [[1, 1], [2, 3], [1, 0]]], dtype=torch.float)
        period = torch.tensor([1, 2], dtype=torch.float).view(2, 1, 1)
        kernel = CosineKernel(batch_shape=torch.Size([2])).initialize(period_length=period)
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = torch.cos(math.pi * ((a[k, i] - b[k, j]) / period[k]).norm(2, dim=-1))

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    for l in range(2):
                        actual[k, l, i, j] = torch.cos(math.pi * ((a[k, i, l] - b[k, j, l]) / period[k]))
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
