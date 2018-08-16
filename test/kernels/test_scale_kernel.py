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
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        base_kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = ScaleKernel(base_kernel)
        kernel.initialize(log_outputscale=torch.Tensor([3]).log())
        kernel.eval()

        actual = torch.Tensor([[16, 4], [4, 0], [64, 36]]).mul_(-0.5).div_(lengthscale ** 2).exp()
        actual = actual * 3
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_forward_batch_mode(self):
        a = torch.Tensor([4, 2, 8]).view(1, 3, 1).repeat(4, 1, 1)
        b = torch.Tensor([0, 2]).view(1, 2, 1).repeat(4, 1, 1)
        lengthscale = 2

        base_kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = ScaleKernel(base_kernel, batch_size=4)
        kernel.initialize(log_outputscale=torch.Tensor([1, 2, 3, 4]).log())
        kernel.eval()

        base_actual = torch.Tensor([[16, 4], [4, 0], [64, 36]]).mul_(-0.5).div_(lengthscale ** 2).exp()
        actual = base_actual.unsqueeze(0).mul(torch.Tensor([1, 2, 3, 4]).view(4, 1, 1))
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)


if __name__ == "__main__":
    unittest.main()
