#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import MaternKernel as GMaternKernel
from gpytorch.kernels.keops import MaternKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

try:
    import pykeops  # noqa

    class TestMatern25KeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return MaternKernel(nu=2.5, **kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return MaternKernel(nu=2.5, ard_num_dims=num_dims, **kwargs)

    class TestMaternKeOpsKernel(unittest.TestCase):
        def forward_x1_eq_x2(self, nu):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()

            kern1 = MaternKernel(nu=nu).cuda()
            kern2 = GMaternKernel(nu=nu).cuda()

            k1 = kern1(x1, x1).evaluate()
            k2 = kern2(x1, x1).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

        def forward_x1_neq_x2(self, nu):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(100, 3).cuda()
            x2 = torch.randn(50, 3).cuda()

            kern1 = MaternKernel(nu=nu).cuda()
            kern2 = GMaternKernel(nu=nu).cuda()

            k1 = kern1(x1, x2).evaluate()
            k2 = kern2(x1, x2).evaluate()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_forward_nu25_x1_eq_x2(self):
            return self.forward_x1_eq_x2(nu=2.5)

        def test_forward_nu25_x1_neq_x2(self):
            return self.forward_x1_neq_x2(nu=2.5)

        def test_forward_nu15_x1_eq_x2(self):
            return self.forward_x1_eq_x2(nu=1.5)

        def test_forward_nu15_x1_neq_x2(self):
            return self.forward_x1_neq_x2(nu=1.5)

        def test_forward_nu05_x1_eq_x2(self):
            return self.forward_x1_eq_x2(nu=0.5)

        def test_forward_nu05_x1_neq_x2(self):
            return self.forward_x1_neq_x2(nu=0.5)

        def test_batch_matmul(self):
            if not torch.cuda.is_available():
                return

            x1 = torch.randn(3, 2, 100, 3).cuda()
            kern1 = MaternKernel(nu=2.5).cuda()
            kern2 = GMaternKernel(nu=2.5).cuda()

            rhs = torch.randn(3, 2, 100, 1).cuda()
            res1 = kern1(x1, x1).matmul(rhs)
            res2 = kern2(x1, x1).matmul(rhs)

            self.assertLess(torch.norm(res1 - res2), 1e-4)


except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
