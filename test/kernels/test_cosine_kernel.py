from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from torch.autograd import Variable
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

        res = kernel(Variable(a), Variable(b)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
