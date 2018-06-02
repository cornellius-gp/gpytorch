from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.kernels import LinearKernel


class TestLinearKernel(unittest.TestCase):
    def test_computes_linear_function_rectangular(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)

        kernel = LinearKernel(num_dimensions=1).initialize(offset=0, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a, b.t())
        res = kernel(Variable(a), Variable(b)).evaluate()
        self.assertLess(torch.norm(res.data - actual), 1e-5)

    def test_computes_linear_function_square(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)

        kernel = LinearKernel(num_dimensions=1).initialize(offset=0, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a, a.t())
        res = kernel(Variable(a), Variable(a)).evaluate()
        self.assertLess(torch.norm(res.data - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
