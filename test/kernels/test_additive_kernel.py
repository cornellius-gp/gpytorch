from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import RBFKernel, AdditiveKernel, ProductKernel


class TestAdditiveKernel(unittest.TestCase):
    def test_computes_product_of_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 * kernel_2

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() ** 2

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 + kernel_2

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() * 2

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_product_of_three_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_3 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = ProductKernel(kernel_1, kernel_2, kernel_3)

        actual = torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float)
        actual = actual.mul_(-0.5).div_(lengthscale ** 2).exp() ** 3

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_of_three_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2], dtype=torch.float).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_3 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = AdditiveKernel(kernel_1, kernel_2, kernel_3)

        actual = (
            torch.tensor([[16, 4], [4, 0], [64, 36]], dtype=torch.float).mul_(-0.5).div_(lengthscale ** 2).exp() * 3
        )

        kernel.eval()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 2e-5)

    def test_computes_sum_radial_basis_function_gradient(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        param = math.log(lengthscale) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / param.exp()) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum() * 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 + kernel_2
        kernel.eval()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.kernels[0].log_lengthscale.grad + kernel.kernels[1].log_lengthscale.grad
        self.assertLess(torch.norm(res - actual_param_grad), 2e-5)

    def test_computes_sum_three_radial_basis_function_gradient(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        param = math.log(lengthscale) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / param.exp()) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum() * 3

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_3 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = AdditiveKernel(kernel_1, kernel_2, kernel_3)
        kernel.eval()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = (
            kernel.kernels[0].log_lengthscale.grad
            + kernel.kernels[1].log_lengthscale.grad
            + kernel.kernels[2].log_lengthscale.grad
        )
        self.assertLess(torch.norm(res - actual_param_grad), 2e-5)


if __name__ == "__main__":
    unittest.main()
