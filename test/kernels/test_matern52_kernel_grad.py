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
                [0.3056225, 0.0000000, -0.5822443, 0.0188260, 0.0209871, -0.0419742],
                [-0.0000000, 0.5822443, 0.0000000, -0.0209871, -0.0056045, 0.0531832],
                [0.5822443, 0.0000000, -0.8515886, 0.0419742, 0.0531832, -0.0853792],
                [0.1304891, 0.2014212, 0.2014212, 0.0336440, 0.0815567, 0.0000000],
                [-0.2014212, -0.1754366, -0.3768578, -0.0815567, -0.1870145, -0.0000000],
                [-0.2014212, -0.3768578, -0.1754366, -0.0000000, -0.0000000, 0.0407784],
            ]
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
