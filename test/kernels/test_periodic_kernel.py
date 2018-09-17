from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import PeriodicKernel


class TestPeriodicKernel(unittest.TestCase):
    def test_computes_periodic_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2
        period = 3
        kernel = PeriodicKernel().initialize(log_lengthscale=math.log(lengthscale), log_period_length=math.log(period))
        kernel.eval()

        actual = torch.zeros(3, 2)
        for i in range(3):
            for j in range(2):
                val = 2 * torch.pow(torch.sin(math.pi * (a[i] - b[j]) / 3), 2) / lengthscale
                actual[i, j] = torch.exp(-val).item()

        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
