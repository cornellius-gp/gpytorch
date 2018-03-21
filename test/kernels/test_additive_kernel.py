from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from torch.autograd import Variable
from gpytorch.kernels import RBFKernel


class TestAdditiveKernel(unittest.TestCase):

    def test_computes_product_of_radial_basis_function(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 * kernel_2
        kernel.eval()
        actual = torch.Tensor([
            [16, 4],
            [4, 0],
            [64, 36],
        ]).mul_(-1).div_(lengthscale).exp() ** 2

        res = kernel(Variable(a), Variable(b)).data
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_sum_of_radial_basis_function(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 + kernel_2
        kernel.eval()
        actual = torch.Tensor([
            [16, 4],
            [4, 0],
            [64, 36],
        ]).mul_(-1).div_(lengthscale).exp() * 2

        res = kernel(Variable(a), Variable(b)).data
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_sum_radial_basis_function_gradient(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2, 2]).view(3, 1)
        lengthscale = 2

        kernel_1 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel_2 = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel = kernel_1 + kernel_2
        kernel.eval()
        param = Variable(
            torch.Tensor(3, 3).fill_(math.log(lengthscale)),
            requires_grad=True,
        )
        diffs = Variable(a.expand(3, 3) - b.expand(3, 3).transpose(0, 1))
        actual_output = (-(diffs ** 2) / (param.exp())).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.data.sum() * 2

        output = kernel(Variable(a), Variable(b))
        output.backward(gradient=torch.eye(3))
        res = kernel.kernel_1.log_lengthscale.grad.data
        res += kernel.kernel_2.log_lengthscale.grad.data
        self.assertLess(torch.norm(res - actual_param_grad), 1e-5)


if __name__ == '__main__':
    unittest.main()
