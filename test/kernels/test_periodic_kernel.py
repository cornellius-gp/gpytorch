from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from torch.autograd import Variable
from gpytorch.kernels import PeriodicKernel


class TestPeriodicKernel(unittest.TestCase):

    def test_computes_periodic_function(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2
        period = 1
        kernel = PeriodicKernel().initialize(
            log_lengthscale=math.log(lengthscale), log_period_length=math.log(period)
        )
        kernel.eval()

        actual = torch.zeros(3, 2)
        for i in range(3):
            for j in range(2):
                val = 2 * torch.pow(torch.sin(math.pi * (a[i] - b[j])), 2) / lengthscale
                actual[i, j] = torch.exp(-val)[0]

        res = kernel(Variable(a), Variable(b)).evaluate().data
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
