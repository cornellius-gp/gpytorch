#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import Matern52KernelGrad
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestMatern52KernelGrad(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return Matern52KernelGrad(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return Matern52KernelGrad(ard_num_dims=num_dims, **kwargs)

    def test_kernel(self, cuda=False):
        a = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3], [0, 4]]], dtype=torch.float)

        actual = torch.tensor(
            [
                [0.3056, -0.0000, 0.5822, 0.0188, -0.0210, 0.0420],
                [0.0000, 0.5822, 0.0000, 0.0210, -0.0056, 0.0532],
                [-0.5822, 0.0000, -0.8516, -0.0420, 0.0532, -0.0854],
                [0.1305, -0.2014, -0.2014, 0.0336, -0.0816, -0.0000],
                [0.2014, -0.1754, -0.3769, 0.0816, -0.1870, -0.0000],
                [0.2014, -0.3769, -0.1754, 0.0000, -0.0000, 0.0408],
            ],
        )

        kernel = Matern52KernelGrad()

        if cuda:
            a = a.cuda()
            b = b.cuda()
            actual = actual.cuda()
            kernel = kernel.cuda()

        res = kernel(a, b).to_dense()

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_kernel_cuda(self):
        if torch.cuda.is_available():
            self.test_kernel(cuda=True)

    def test_kernel_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)

        kernel = Matern52KernelGrad()
        res = kernel(a, b).to_dense()

        # Compute each batch separately
        actual = torch.zeros(2, 8, 8)
        actual[0, :, :] = kernel(a[0, :, :].squeeze(), b[0, :, :].squeeze()).to_dense()
        actual[1, :, :] = kernel(a[1, :, :].squeeze(), b[1, :, :].squeeze()).to_dense()

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = Matern52KernelGrad()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = Matern52KernelGrad(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
