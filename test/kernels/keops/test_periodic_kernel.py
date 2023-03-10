#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.kernels import PeriodicKernel as GPeriodicKernel
from gpytorch.kernels.keops import PeriodicKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

try:
    import pykeops  # noqa

    class TestPeriodicKeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return PeriodicKernel(**kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return PeriodicKernel(ard_num_dims=num_dims, **kwargs)

    class TestPeriodicKeOpsKernel(unittest.TestCase):

        # tests the keops implementation
        def test_forward_x1_eq_x2(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(2):

                x1 = torch.randn(100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()

                k1 = kern1(x1, x1).to_dense()
                k2 = kern2(x1, x1).to_dense()

                self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_forward_x1_neq_x2(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(2):

                x1 = torch.randn(100, 3).cuda()
                x2 = torch.randn(100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()

                k1 = kern1(x1, x2).to_dense()
                k2 = kern2(x1, x2).to_dense()

                self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_batch_matmul(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(2):

                x1 = torch.randn(3, 2, 100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()
                rhs = torch.randn(3, 2, 100, 1).cuda()

                res1 = kern1(x1, x1).matmul(rhs)
                res2 = kern2(x1, x1).matmul(rhs)

                self.assertLess(torch.norm(res1 - res2), 1e-4)

        # tests the nonkeops implementation (_nonkeops_covar_func)
        def test_forward_x1_eq_x2_nonkeops(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(800):

                x1 = torch.randn(100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()

                k1 = kern1(x1, x1).to_dense()
                k2 = kern2(x1, x1).to_dense()

            self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_forward_x1_neq_x2_nonkeops(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(800):

                x1 = torch.randn(100, 3).cuda()
                x2 = torch.randn(100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()
                k1 = kern1(x1, x2).to_dense()
                k2 = kern2(x1, x2).to_dense()

                self.assertLess(torch.norm(k1 - k2), 1e-4)

        def test_batch_matmul_nonkeops(self):
            if not torch.cuda.is_available():
                return

            with gpytorch.settings.max_cholesky_size(800):

                x1 = torch.randn(3, 2, 100, 3).cuda()
                kern1 = PeriodicKernel().cuda()
                kern2 = GPeriodicKernel().cuda()
                rhs = torch.randn(3, 2, 100, 1).cuda()
                res1 = kern1(x1, x1).matmul(rhs)
                res2 = kern2(x1, x1).matmul(rhs)

                self.assertLess(torch.norm(res1 - res2), 1e-4)

    def create_kernel_with_prior(self, period_length_prior):
        return PeriodicKernel(period_length_prior=period_length_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        kernel_fn = lambda prior: self.create_kernel_with_prior(prior)
        kernel_fn(None)
        kernel_fn(NormalPrior(0, 1))
        self.assertRaises(TypeError, kernel_fn, 1)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
