#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import LinearKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestLinearKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return LinearKernel(**kwargs)

    def test_computes_linear_function_rectangular(self):
        a = torch.tensor([4, 2, 8], dtype=torch.float).view(3, 1)
        b = torch.tensor([0, 2, 1], dtype=torch.float).view(3, 1)

        kernel = LinearKernel().initialize(variance=1.0)
        kernel.eval()
        actual = torch.matmul(a, b.t())
        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_computes_linear_function_square(self):
        a = torch.tensor([[4, 1], [2, 0], [8, 3]], dtype=torch.float)

        kernel = LinearKernel().initialize(variance=3.14)
        kernel.eval()
        actual = torch.matmul(a, a.t()) * 3.14
        res = kernel(a, a).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, a).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims
        dim_group_a = a
        dim_group_a = dim_group_a.permute(1, 0).reshape(-1, 3)
        actual = 3.14 * torch.mul(dim_group_a.unsqueeze(-1), dim_group_a.unsqueeze(-2))
        res = kernel(a, a, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims + diag
        res = kernel(a, a, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-4)

    def test_computes_linear_function_square_batch(self):
        a = torch.tensor([[[4, 1], [2, 0], [8, 3]], [[1, 1], [2, 1], [1, 3]]], dtype=torch.float)

        kernel = LinearKernel().initialize(variance=1.0)
        kernel.eval()
        actual = torch.matmul(a, a.transpose(-1, -2))
        res = kernel(a, a).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # diag
        res = kernel(a, a).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims
        dim_group_a = a
        dim_group_a = dim_group_a.transpose(-1, -2).unsqueeze(-1)
        actual = dim_group_a.matmul(dim_group_a.transpose(-2, -1))
        res = kernel(a, a, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-4)

        # batch_dims + diag
        res = kernel(a, a, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-2, dim2=-1)
        self.assertLess(torch.norm(res - actual), 1e-4)

    def create_kernel_with_prior(self, variance_prior):
        return self.create_kernel_no_ard(variance_prior=variance_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        self.create_kernel_with_prior(None)
        self.create_kernel_with_prior(NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)


if __name__ == "__main__":
    unittest.main()
