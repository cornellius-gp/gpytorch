from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import RBFKernel, ScaleKernel


class TestScaleKernel(unittest.TestCase):
    def test_forward(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        base_kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = ScaleKernel(base_kernel)
        kernel.initialize(log_outputscale=torch.tensor([3], dtype=torch.float).log())
        kernel.eval()

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual.mul_(-0.5).div_(lengthscale ** 2).exp_().mul_(3)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_forward_batch_mode(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(1, 3, 1).repeat(4, 1, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(1, 2, 1).repeat(4, 1, 1)
        lengthscale = 2

        base_kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = ScaleKernel(base_kernel, batch_size=4)
        kernel.initialize(log_outputscale=torch.tensor([1, 2, 3, 4], dtype=torch.float).log())
        kernel.eval()

        base_actual = (
            torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float).mul_(-0.5).div_(lengthscale ** 2).exp()
        )
        actual = base_actual.unsqueeze(0).mul(torch.tensor([1, 2, 3, 4], dtype=torch.float).view(4, 1, 1))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
