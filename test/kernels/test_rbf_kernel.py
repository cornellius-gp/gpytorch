from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import unittest
from gpytorch.kernels import RBFKernel


class TestRBFKernel(unittest.TestCase):
    def test_ard(self):
        a = torch.Tensor([[1, 2], [2, 4]])
        b = torch.Tensor([1, 3]).view(1, 1, 2)
        lengthscales = torch.Tensor([1, 2]).view(1, 1, 2)

        kernel = RBFKernel(ard_num_dims=2)
        kernel.initialize(log_lengthscale=lengthscales.log())
        kernel.eval()

        actual = (a - b).div_(lengthscales).pow(2).sum(dim=-1).mul_(-0.5).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual.unsqueeze(-1)), 1e-5)

    def test_subset_active_compute_radial_basis_function(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        a_p = torch.Tensor([1, 2, 3]).view(3, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel = RBFKernel(active_dims=[0])
        kernel.initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.Tensor([[16, 4], [4, 0], [64, 36]]).mul_(-0.5).div_(lengthscale ** 2).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_radial_basis_function(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2]).view(2, 1)
        lengthscale = 2

        kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.Tensor([[16, 4], [4, 0], [64, 36]]).mul_(-0.5).div_(lengthscale ** 2).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_forward_diag(self):
        a = torch.Tensor([4, 2, 8]).view(1, 3, 1)
        b = torch.Tensor([2, 0, 6]).view(1, 3, 1)
        lengthscale = 2

        kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        res = kernel.forward_diag(a, b).squeeze()
        actual = torch.Tensor([0.60653066, 0.60653066, 0.60653066])

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_radial_basis_function_gradient(self):
        a = torch.Tensor([4, 2, 8]).view(3, 1)
        b = torch.Tensor([0, 2, 2]).view(3, 1)
        lengthscale = 2

        kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        param = math.log(lengthscale) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / param.exp()) ** 2).exp()
        actual_output.backward(gradient=torch.eye(3))
        actual_param_grad = param.grad.sum()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.log_lengthscale.grad

        self.assertLess(torch.norm(res - actual_param_grad), 1e-5)

    def test_subset_active_computes_radial_basis_function_gradient(self):
        a_1 = torch.Tensor([4, 2, 8]).view(3, 1)
        a_p = torch.Tensor([1, 2, 3]).view(3, 1)
        a = torch.cat((a_1, a_p), 1)
        b = torch.Tensor([0, 2, 2]).view(3, 1)
        lengthscale = 2

        param = math.log(lengthscale) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a_1.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / param.exp()) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum()

        kernel = RBFKernel(active_dims=[0])
        kernel.initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()
        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.log_lengthscale.grad

        self.assertLess(torch.norm(res - actual_param_grad), 1e-5)


if __name__ == "__main__":
    unittest.main()
