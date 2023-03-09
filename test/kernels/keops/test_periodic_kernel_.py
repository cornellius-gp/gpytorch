#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import PeriodicKernel as GPeriodicKernel
from gpytorch.kernels.keops import PeriodicKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase
from gpytorch import settings

try:
    import pykeops  # noqa

    class TestPeriodicKeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return PeriodicKernel(**kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return PeriodicKernel(ard_num_dims=num_dims, **kwargs)

    class TestPeriodicKeOpsKernel(unittest.TestCase):
        def test_forward_x1_eq_x2(self):

            x1 = torch.randn(501, 3)

            kern1 = PeriodicKernel()
            kern2 = GPeriodicKernel()

            k1 = kern1(x1, x1).to_dense()
            k2 = kern2(x1, x1).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_forward_x1_neq_x2(self):

            x1 = torch.randn(501, 3)
            x2 = torch.randn(500, 3)

            kern1 = PeriodicKernel()
            kern2 = GPeriodicKernel()

            k1 = kern1(x1, x2).to_dense()
            k2 = kern2(x1, x2).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_batch_matmul(self):

            x1 = torch.randn(3, 2, 100, 3)
            kern1 = PeriodicKernel()
            kern2 = GPeriodicKernel()

            rhs = torch.randn(3, 2, 100, 1)
            res1 = kern1(x1, x1).matmul(rhs)
            res2 = kern2(x1, x1).matmul(rhs)

            self.assertLess(torch.norm(res1 - res2), 1e-4)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
