#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import RBFKernelGrad
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestRBFKernelGrad(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return RBFKernelGrad(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return RBFKernelGrad(ard_num_dims=num_dims, **kwargs)

    def test_kernel(self, cuda=False):
        a = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3], [0, 4]]], dtype=torch.float)

        actual = torch.tensor(
            [
                [0.35321, 0, -0.73517, 0.0054977, 0.011443, -0.022886],
                [0, 0.73517, 0, -0.011443, -0.012374, 0.047633],
                [0.73517, 0, -0.79499, 0.022886, 0.047633, -0.083824],
                [0.12476, 0.25967, 0.25967, 0.015565, 0.064793, 0],
                [-0.25967, -0.2808, -0.54047, -0.064793, -0.23732, 0],
                [-0.25967, -0.54047, -0.2808, 0, 0, 0.032396],
            ]
        )

        kernel = RBFKernelGrad()

        if cuda:
            a = a.cuda()
            b = b.cuda()
            actual = actual.cuda()
            kernel = kernel.cuda()

        res = kernel(a, b).evaluate()

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_kernel_cuda(self):
        if torch.cuda.is_available():
            self.test_kernel(cuda=True)

    def test_kernel_batch(self):
        a = torch.tensor([[[1, 2, 3], [2, 4, 0]], [[-1, 1, 2], [2, 1, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3, 1]], [[2, -1, 0]]], dtype=torch.float).repeat(1, 2, 1)

        kernel = RBFKernelGrad()
        res = kernel(a, b).evaluate()

        # Compute each batch separately
        actual = torch.zeros(2, 8, 8)
        actual[0, :, :] = kernel(a[0, :, :].squeeze(), b[0, :, :].squeeze()).evaluate()
        actual[1, :, :] = kernel(a[1, :, :].squeeze(), b[1, :, :].squeeze()).evaluate()

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = RBFKernelGrad()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = RBFKernelGrad(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
