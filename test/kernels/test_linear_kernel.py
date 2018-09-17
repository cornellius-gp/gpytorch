from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.kernels import LinearKernel


class TestLinearKernel(unittest.TestCase):
    def test_computes_linear_function_rectangular(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)

        kernel = LinearKernel(num_dimensions=1).initialize(offset=0, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a, b.t())
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_linear_function_square(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)

        kernel = LinearKernel(num_dimensions=1).initialize(offset=0, variance=1.0)
        kernel.eval()
        actual = 1 + torch.matmul(a, a.t())
        res = kernel(a, a).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
