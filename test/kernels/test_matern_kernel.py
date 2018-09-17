from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import MaternKernel


class TestMaternKernel(unittest.TestCase):
    def test_forward_nu_1_over_2(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=0.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.Tensor([[4, 2], [2, 0], [8, 6]]).div_(-lengthscale).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_3_over_2(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=1.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        dist = torch.Tensor([[4, 2], [2, 0], [8, 6]]).mul_(math.sqrt(3) / lengthscale)
        actual = (dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)

    def test_forward_nu_5_over_2(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel = MaternKernel(nu=2.5).initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        dist = torch.Tensor([[4, 2], [2, 0], [8, 6]]).mul_(math.sqrt(5) / lengthscale)
        actual = (dist ** 2 / 3 + dist + 1).mul(torch.exp(-dist))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-3)


if __name__ == "__main__":
    unittest.main()
