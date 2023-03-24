#!/usr/bin/env python3

import unittest

from gpytorch.kernels import MaternKernel as GMaternKernel
from gpytorch.kernels.keops import MaternKernel
from gpytorch.test.base_keops_test_case import BaseKeOpsTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

try:
    import pykeops  # noqa

    class TestMatern25KeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return MaternKernel(nu=2.5, **kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return MaternKernel(nu=2.5, ard_num_dims=num_dims, **kwargs)

    class TestMaternKeOpsKernel(unittest.TestCase, BaseKeOpsTestCase):
        @property
        def k1(self):
            return MaternKernel

        @property
        def k2(self):
            return GMaternKernel

        def test_forward_nu25_x1_eq_x2(self):
            return self.test_forward_x1_eq_x2(nu=2.5)

        def test_forward_nu25_x1_neq_x2(self):
            return self.test_forward_x1_neq_x2(nu=2.5)

        def test_forward_nu15_x1_eq_x2(self):
            return self.test_forward_x1_eq_x2(nu=1.5)

        def test_forward_nu15_x1_neq_x2(self):
            return self.test_forward_x1_neq_x2(nu=1.5)

        def test_forward_nu05_x1_eq_x2(self):
            return self.test_forward_x1_eq_x2(nu=0.5)

        def test_forward_nu05_x1_neq_x2(self):
            return self.test_forward_x1_neq_x2(nu=0.5)

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
