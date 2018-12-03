#!/usr/bin/env python3

import torch
import unittest
from gpytorch.kernels import RBFKernel, ScaleKernel


class TestScaleKernel(unittest.TestCase):
    def test_ard(self):
        a = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float).repeat(2, 1, 1)
        b = torch.tensor([[[1, 3], [0, 4]]], dtype=torch.float).repeat(2, 1, 1)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 1, 2)

        base_kernel = RBFKernel(ard_num_dims=2)
        base_kernel.initialize(log_lengthscale=lengthscales.log())
        kernel = ScaleKernel(base_kernel)
        kernel.initialize(log_outputscale=torch.tensor([3], dtype=torch.float).log())
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        actual.mul_(3)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # Diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp().view(4, 2, 2)
        actual.mul_(3)
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

        base_kernel = RBFKernel(batch_size=2, ard_num_dims=3)
        base_kernel.initialize(log_lengthscale=lengthscales.log())
        kernel = ScaleKernel(base_kernel, batch_size=2)
        kernel.initialize(log_outputscale=torch.tensor([1, 2], dtype=torch.float).log())
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        actual[1].mul_(2)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp().view(6, 2, 2)
        actual[3:].mul_(2)
        res = kernel(a, b, batch_dims=(0, 2)).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, batch_dims=(0, 2)).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_outputscale(self):
        kernel = ScaleKernel(RBFKernel())
        kernel.initialize(outputscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.outputscale)
        self.assertLess(torch.norm(kernel.outputscale - actual_value), 1e-5)

    def test_initialize_outputscale_batch(self):
        kernel = ScaleKernel(RBFKernel(), batch_size=2)
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(outputscale=ls_init)
        actual_value = ls_init.view_as(kernel.outputscale)
        self.assertLess(torch.norm(kernel.outputscale - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
