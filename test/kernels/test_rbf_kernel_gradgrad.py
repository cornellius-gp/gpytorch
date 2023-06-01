#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import RBFKernelGradGrad
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestRBFKernelGradGrad(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return RBFKernelGradGrad(**kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return RBFKernelGradGrad(ard_num_dims=num_dims, **kwargs)

    def test_kernel(self, cuda=False):
        a = torch.tensor([[[1, 2], [2, 4]]], dtype=torch.float)
        b = torch.tensor([[[1, 3], [0, 4]]], dtype=torch.float)

        actual = torch.tensor(
            [
                [
                    [
                        3.5321289e-01,
                        0.0000000e00,
                        -7.3516625e-01,
                        -7.3516631e-01,
                        7.9498571e-01,
                        5.4977159e-03,
                        1.1442775e-02,
                        -2.2885550e-02,
                        1.2373861e-02,
                        8.3823770e-02,
                    ],
                    [
                        -0.0000000e00,
                        7.3516631e-01,
                        0.0000000e00,
                        0.0000000e00,
                        -0.0000000e00,
                        -1.1442775e-02,
                        -1.2373861e-02,
                        4.7633272e-02,
                        2.1878703e-02,
                        -1.7446819e-01,
                    ],
                    [
                        7.3516625e-01,
                        0.0000000e00,
                        -7.9498571e-01,
                        -1.5301522e00,
                        -1.4056460e00,
                        2.2885550e-02,
                        4.7633272e-02,
                        -8.3823770e-02,
                        5.1509142e-02,
                        2.5366980e-01,
                    ],
                    [
                        -7.3516631e-01,
                        -0.0000000e00,
                        1.5301522e00,
                        4.5904574e00,
                        -1.6546586e00,
                        1.2373861e-02,
                        -2.1878703e-02,
                        -5.1509142e-02,
                        -1.2280136e-01,
                        1.8866448e-01,
                    ],
                    [
                        7.9498571e-01,
                        0.0000000e00,
                        1.4056460e00,
                        -1.6546586e00,
                        -7.8896437e00,
                        8.3823770e-02,
                        1.7446819e-01,
                        -2.5366980e-01,
                        1.8866447e-01,
                        5.3255635e-01,
                    ],
                    [
                        1.2475928e-01,
                        2.5967008e-01,
                        2.5967011e-01,
                        2.8079915e-01,
                        2.8079927e-01,
                        1.5564885e-02,
                        6.4792536e-02,
                        0.0000000e00,
                        2.3731807e-01,
                        -3.2396268e-02,
                    ],
                    [
                        -2.5967008e-01,
                        -2.8079915e-01,
                        -5.4046929e-01,
                        4.9649185e-01,
                        -5.8444691e-01,
                        -6.4792536e-02,
                        -2.3731807e-01,
                        0.0000000e00,
                        -7.1817851e-01,
                        1.3485716e-01,
                    ],
                    [
                        -2.5967011e-01,
                        -5.4046929e-01,
                        -2.8079927e-01,
                        -5.8444673e-01,
                        4.9649167e-01,
                        -0.0000000e00,
                        0.0000000e00,
                        3.2396268e-02,
                        0.0000000e00,
                        0.0000000e00,
                    ],
                    [
                        2.8079915e-01,
                        -4.9649185e-01,
                        5.8444673e-01,
                        -2.7867227e00,
                        6.3200271e-01,
                        2.3731807e-01,
                        7.1817851e-01,
                        0.0000000e00,
                        1.5077497e00,
                        -4.9394643e-01,
                    ],
                    [
                        2.8079927e-01,
                        5.8444691e-01,
                        -4.9649167e-01,
                        6.3200271e-01,
                        -2.7867231e00,
                        -3.2396268e-02,
                        -1.3485716e-01,
                        -0.0000000e00,
                        -4.9394643e-01,
                        2.0228577e-01,
                    ],
                ]
            ]
        )

        kernel = RBFKernelGradGrad()

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

        kernel = RBFKernelGradGrad()
        res = kernel(a, b).to_dense()

        # Compute each batch separately
        actual = torch.zeros(2, 14, 14)
        actual[0, :, :] = kernel(a[0, :, :].squeeze(), b[0, :, :].squeeze()).to_dense()
        actual[1, :, :] = kernel(a[1, :, :].squeeze(), b[1, :, :].squeeze()).to_dense()

        self.assertLess(torch.norm(res - actual), 1e-5)

    def test_initialize_lengthscale(self):
        kernel = RBFKernelGradGrad()
        kernel.initialize(lengthscale=3.14)
        actual_value = torch.tensor(3.14).view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)

    def test_initialize_lengthscale_batch(self):
        kernel = RBFKernelGradGrad(batch_shape=torch.Size([2]))
        ls_init = torch.tensor([3.14, 4.13])
        kernel.initialize(lengthscale=ls_init)
        actual_value = ls_init.view_as(kernel.lengthscale)
        self.assertLess(torch.norm(kernel.lengthscale - actual_value), 1e-5)


if __name__ == "__main__":
    unittest.main()
