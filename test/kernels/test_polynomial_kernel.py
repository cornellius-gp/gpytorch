#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import PolynomialKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PolynomialKernel(power=2, **kwargs)

    def test_computes_quadratic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=2)
        kernel.eval()

        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[i, j] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)
        # diag
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[:, i].unsqueeze(-1), b[:, i].unsqueeze(-1)).to_dense()

        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_computes_cubic_kernel(self):
        a = torch.tensor([[4, 1], [2, 2], [8, 0]], dtype=torch.float)
        b = torch.tensor([[0, 0], [2, 1], [1, 0]], dtype=torch.float)
        kernel = PolynomialKernel(power=3)
        kernel.eval()

        actual = torch.zeros(3, 3)
        for i in range(3):
            for j in range(3):
                actual[i, j] = (a[i].matmul(b[j]) + kernel.offset).pow(kernel.power)

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # diag
        res = kernel(a, b).diagonal(dim1=-1, dim2=-2)
        actual = actual.diagonal(dim1=-1, dim2=-2)
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims
        actual = torch.zeros(2, 3, 3)
        for i in range(2):
            actual[i] = kernel(a[:, i].unsqueeze(-1), b[:, i].unsqueeze(-1)).to_dense()

        res = kernel(a, b, last_dim_is_batch=True).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

        # batch_dims + diag
        res = kernel(a, b, last_dim_is_batch=True).diagonal(dim1=-1, dim2=-2)
        actual = torch.cat([actual[i].diagonal(dim1=-1, dim2=-2).unsqueeze(0) for i in range(actual.size(0))])
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_quadratic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=2, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = (a[k, i].matmul(b[k, j]) + kernel.offset[k]).pow(kernel.power)

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_cubic_kernel_batch(self):
        a = torch.tensor([[4, 2, 8], [1, 2, 3]], dtype=torch.float).view(2, 3, 1)
        b = torch.tensor([[0, 2, 1], [-1, 2, 0]], dtype=torch.float).view(2, 3, 1)
        kernel = PolynomialKernel(power=3, batch_shape=torch.Size([2])).initialize(offset=torch.rand(2, 1))
        kernel.eval()

        actual = torch.zeros(2, 3, 3)
        for k in range(2):
            for i in range(3):
                for j in range(3):
                    actual[k, i, j] = (a[k, i].matmul(b[k, j]) + kernel.offset[k]).pow(kernel.power)

        res = kernel(a, b).to_dense()
        self.assertLess(torch.norm(res - actual), 1e-5)

    def create_kernel_with_prior(self, offset_prior):
        return self.create_kernel_no_ard(offset_prior=offset_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        self.create_kernel_with_prior(None)
        self.create_kernel_with_prior(NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1)


if __name__ == "__main__":
    unittest.main()
