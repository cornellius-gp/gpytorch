#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import LinearKernel, RBFKernel, ScaleKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestScaleKernel(BaseKernelTestCase, unittest.TestCase):
    def create_kernel_no_ard(self, **kwargs):
        base_kernel = RBFKernel()
        kernel = ScaleKernel(base_kernel, **kwargs)
        return kernel

    def create_kernel_ard(self, num_dims, **kwargs):
        base_kernel = RBFKernel(ard_num_dims=num_dims)
        kernel = ScaleKernel(base_kernel, **kwargs)
        return kernel

    def test_ard(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        b = torch.tensor([[1, 3], [0, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 2)

        base_kernel = RBFKernel(ard_num_dims=2)
        base_kernel.initialize(lengthscale=lengthscales)
        kernel = ScaleKernel(base_kernel)
        kernel.initialize(outputscale=torch.tensor([3], dtype=torch.float))
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        actual = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1).mul_(-0.5).exp()
        actual.mul_(3)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # Diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp().view(2, 2, 2)
        actual.mul_(3)
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = torch.cat([actual[i].diag().unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        base_kernel = RBFKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        base_kernel.initialize(lengthscale=lengthscales)
        kernel = ScaleKernel(base_kernel, batch_shape=torch.Size([2]))
        kernel.initialize(outputscale=torch.tensor([1, 2], dtype=torch.float))
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
        double_batch_a = scaled_a.transpose(-1, -2)
        double_batch_b = scaled_b.transpose(-1, -2)
        actual = double_batch_a.unsqueeze(-1) - double_batch_b.unsqueeze(-2)
        actual = actual.pow(2).mul_(-0.5).exp()
        actual[1, :, :, :].mul_(2)
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_outputscale(self):
        kernel = ScaleKernel(RBFKernel())
        kernel.initialize(outputscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.outputscale)
        self.assertLess(torch.norm(kernel.outputscale - actual_value), 1e-5)

    def test_initialize_outputscale_batch(self):
        kernel = ScaleKernel(RBFKernel(), batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(outputscale=ls_init)
        actual_value = ls_init.view_as(kernel.outputscale)
        self.assertLess(torch.norm(kernel.outputscale - actual_value), 1e-5)

    def test_stationary(self):
        kernel = ScaleKernel(RBFKernel())
        self.assertTrue(kernel.is_stationary)

    def test_non_stationary(self):
        kernel = ScaleKernel(LinearKernel())
        self.assertFalse(kernel.is_stationary)

    def test_inherit_active_dims(self):
        lengthscales = torch.tensor([1, 1], dtype=torch.float)
        base_kernel = RBFKernel(active_dims=(1, 2), ard_num_dims=2)
        base_kernel.initialize(lengthscale=lengthscales)
        kernel = ScaleKernel(base_kernel)
        kernel.initialize(outputscale=torch.tensor([3], dtype=torch.float))
        kernel.eval()
        self.assertTrue(torch.all(kernel.active_dims == base_kernel.active_dims))


if __name__ == "__main__":
    unittest.main()
