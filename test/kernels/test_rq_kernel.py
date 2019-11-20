#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import RQKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestRQKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return RQKernel(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return RQKernel(ard_num_dims=num_dims, **kwargs)

    def test_ard(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        b = torch.tensor([[1, 3], [0, 4]], dtype=torch.float)
        lengthscales = torch.tensor([1, 2], dtype=torch.float).view(1, 2)

        kernel = RQKernel(ard_num_dims=2)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1)
        actual = dist.div_(2 * kernel.alpha).add_(1.0).pow(-kernel.alpha)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # Diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        diff = scaled_a.transpose(-1, -2).unsqueeze(-1) - scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = diff.pow(2).div_(2 * kernel.alpha).add_(1.0).pow(-kernel.alpha)
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]]], dtype=torch.float)

        kernel = RQKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1)
        actual = dist.div_(2 * kernel.alpha).add_(1.0).pow(-kernel.alpha)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

        # # batch_dims
        double_batch_a = scaled_a.transpose(-1, -2).unsqueeze(-1)
        double_batch_b = scaled_b.transpose(-1, -2).unsqueeze(-2)
        actual = double_batch_a - double_batch_b
        alpha = kernel.alpha.view(2, 1, 1, 1)
        actual = actual.pow_(2).div_(2 * alpha).add_(1.0).pow(-alpha)
        res = kernel(a, b, last_dim_is_batch=True).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims and diag
        res = kernel(a, b, last_dim_is_batch=True).diag()
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_ard_separate_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)
        lengthscales = torch.tensor([[[1, 2, 1]], [[2, 1, 0.5]]], dtype=torch.float)

        kernel = RQKernel(batch_shape=torch.Size([2]), ard_num_dims=3)
        kernel.initialize(lengthscale=lengthscales)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        scaled_a = a.div(lengthscales)
        scaled_b = b.div(lengthscales)
        dist = (scaled_a.unsqueeze(-2) - scaled_b.unsqueeze(-3)).pow(2).sum(dim=-1)
        actual = dist.div_(2 * kernel.alpha).add_(1.0).pow(-kernel.alpha)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_rational_quadratic(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2

        kernel = RQKernel().initialize(lengthscale=lengthscale)
        kernel.eval()

        dist = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float).div(lengthscale ** 2)
        actual = dist.div_(2 * kernel.alpha).add_(1.0).pow(-kernel.alpha)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_rational_quadratic_gradient(self):
        softplus = torch.nn.functional.softplus
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        kernel = RQKernel()
        kernel.initialize(lengthscale=2.0)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        raw_lengthscale = torch.tensor(math.log(math.exp(2.0) - 1))
        raw_lengthscale.requires_grad_()
        raw_alpha = torch.tensor(math.log(math.exp(3.0) - 1))
        raw_alpha.requires_grad_()
        lengthscale, alpha = softplus(raw_lengthscale), softplus(raw_alpha)
        dist = (a.expand(3, 3) - b.expand(3, 3).transpose(0, 1)).div(lengthscale).pow(2)
        actual_output = dist.div(2 * alpha).add(1).pow(-alpha)
        actual_output.backward(gradient=torch.eye(3))

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))

        res = kernel.raw_lengthscale.grad
        self.assertLess(torch.norm(res - raw_lengthscale.grad), 1e-5)
        res = kernel.raw_alpha.grad
        self.assertLess(torch.norm(res - raw_alpha.grad), 1e-5)

    def test_subset_active_compute_rational_quadratic(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a, a_p), 1)
        b = torch.tensor([0, 2, 4], dtype=torch.float).view(3, 1)
        lengthscale = 2

        kernel = RQKernel(active_dims=[0])
        kernel.initialize(lengthscale=lengthscale)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        actual = torch.tensor([[16, 4, 0], [4, 0, 4], [64, 36, 16]], dtype=torch.float)
        actual.div_(lengthscale ** 2).div_(2 * kernel.alpha).add_(1).pow_(-kernel.alpha)
        res = kernel(a, b).evaluate()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diag()
        actual = actual.diag()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_subset_active_computes_rational_quadratic_gradient(self):
        softplus = torch.nn.functional.softplus
        a_1 = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        a_p = torch.tensor([1, 2, 3], dtype=torch.float).view(3, 1)
        a = torch.cat((a_1, a_p), 1)
        b = torch.tensor([0, 2, 2], dtype=torch.float).view(3, 1)

        kernel = RQKernel(active_dims=[0])
        kernel.initialize(lengthscale=2.0)
        kernel.initialize(alpha=3.0)
        kernel.eval()

        raw_lengthscale = torch.tensor(math.log(math.exp(2.0) - 1))
        raw_lengthscale.requires_grad_()
        raw_alpha = torch.tensor(math.log(math.exp(3.0) - 1))
        raw_alpha.requires_grad_()
        lengthscale, alpha = softplus(raw_lengthscale), softplus(raw_alpha)
        dist = (a_1.expand(3, 3) - b.expand(3, 3).transpose(0, 1)).div(lengthscale).pow(2)
        actual_output = dist.div(2 * alpha).add(1).pow(-alpha)
        actual_output.backward(gradient=torch.eye(3))

        output = kernel(a, b).evaluate()
        output.backward(gradient=torch.eye(3))

        res = kernel.raw_lengthscale.grad
        self.assertLess(torch.norm(res - raw_lengthscale.grad), 1e-5)
        res = kernel.raw_alpha.grad
        self.assertLess(torch.norm(res - raw_alpha.grad), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = RQKernel()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = RQKernel(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_alpha(self):
        kernel = RQKernel()
        kernel.initialize(alpha=3.0)
        actual_value = torch.tensor(3.0).view_as(kernel.alpha)
        self.assertLess(torch.norm(kernel.alpha - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
