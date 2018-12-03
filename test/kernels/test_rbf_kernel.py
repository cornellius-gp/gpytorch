#!/usr/bin/env python3

import math
import torch
import unittest
from gpytorch.kernels import RBFKernel


class TestRBFKernel(unittest.TestCase):
    def test_ard(self):
        a = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3], [0, 4]]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        kernel = RBFKernel(ard_num_dims=2)
        kernel.initialize(log_lengthscale=lengthscales.log())
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # Diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp().view(2, 2, 2)
        res = kernel(a, b, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = RBFKernel(batch_size=2, ard_num_dims=3)
        kernel.initialize(log_lengthscale=lengthscales.log())
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp().view(6, 2, 2)
        res = kernel(a, b, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = RBFKernel(batch_size=2, ard_num_dims=3)
        kernel.initialize(log_lengthscale=lengthscales.log())
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_subset_active_compute_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2

        kernel = RBFKernel(active_dims=[0])
        kernel.initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float)
        actual.mul_(-0.5).div_(lengthscale ** 2).exp_()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_radial_basis_function(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2

        kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        actual = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float)
        actual.mul_(-0.5).div_(lengthscale ** 2).exp_()
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_radial_basis_function_gradient(self):
        softplus = torch.nn.functional.softplus
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        kernel = RBFKernel().initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()

        param = math.log(math.exp(lengthscale) - 1) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / softplus(param)) ** 2).exp()
        actual_output.backward(gradient=torch.eye(3))
        actual_param_grad = param.grad.sum()

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.raw_lengthscale.grad

        self.assertLess(torch.norm(res - actual_param_grad), 1e-5)

    def test_subset_active_computes_radial_basis_function_gradient(self):
        softplus = torch.nn.functional.softplus
        a_1 = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a_1, a_p), 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)
        lengthscale = 2

        param = math.log(math.exp(lengthscale) - 1) * torch.ones(3, 3)
        param.requires_grad_()
        diffs = a_1.expand(3, 3) - b.expand(3, 3).transpose(0, 1)
        actual_output = (-0.5 * (diffs / softplus(param)) ** 2).exp()
        actual_output.backward(torch.eye(3))
        actual_param_grad = param.grad.sum()

        kernel = RBFKernel(active_dims=[0])
        kernel.initialize(log_lengthscale=math.log(lengthscale))
        kernel.eval()
        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))
        res = kernel.raw_lengthscale.grad

        self.assertLess(torch.norm(res - actual_param_grad), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = RBFKernel()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = RBFKernel(batch_size=2)
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
