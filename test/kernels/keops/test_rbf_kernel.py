#!/usr/bin/env python3

import unittest

from gpytorch.kernels import RBFKernel as GRBFKernel
from gpytorch.kernels.keops import RBFKernel
from gpytorch.test.base_keops_test_case import BaseKeOpsTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

try:
    import pykeops  # noqa

    class TestRBFKeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return RBFKernel(**kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return RBFKernel(ard_num_dims=num_dims, **kwargs)

    class TestRBFKeOpsKernel(unittest.TestCase, BaseKeOpsTestCase):
        @property
        def k1(self):
            return RBFKernel

        @property
        def k2(self):
            return GRBFKernel

except ImportError:
    pass

if __name__ == "__main__":
    unittest.main()
