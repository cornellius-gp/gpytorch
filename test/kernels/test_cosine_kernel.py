from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import CosineKernel


class TestCosineKernel(unittest.TestCase):
    def test_computes_periodic_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        period = 1
        kernel = CosineKernel().initialize(log_period_length=math.log(period))
        kernel.eval()

        actual = torch.zeros(3, 2)
        for i in range(3):
            for j in range(2):
                actual[i, j] = torch.cos(math.pi * (a[i] - b[j]) / period)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2], [-1, 2]], dtype=torch.float).view(2, 2, 1)
        period = torch.tensor(1, dtype=torch.float).view(1, 1, 1)
        kernel = CosineKernel().initialize(log_period_length=torch.log(period))
        kernel.eval()

        actual = torch.zeros(2, 3, 2)
        for k in range(2):
            for i in range(3):
                for j in range(2):
                    actual[k, i, j] = torch.cos(math.pi * (a[k, i] - b[k, j]) / period)

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_batch_separate(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2], [-1, 2]], dtype=torch.float).view(2, 2, 1)
        period = torch.tensor([1, 2], dtype=torch.float).view(2, 1, 1)
        kernel = CosineKernel(batch_size=2).initialize(log_period_length=torch.log(period))
        kernel.eval()

        actual = torch.zeros(2, 3, 2)
        for k in range(2):
            for i in range(3):
                for j in range(2):
                    actual[k, i, j] = torch.cos(math.pi * (a[k, i] - b[k, j]) / period[k])

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
