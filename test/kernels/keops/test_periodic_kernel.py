#!/usr/bin/env python3

import unittest

from gpytorch.kernels import PeriodicKernel as GPeriodicKernel
from gpytorch.kernels.keops import PeriodicKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_keops_test_case import BaseKeOpsTestCase
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

try:
    import pykeops  # noqa

    class TestPeriodicKeOpsBaseKernel(unittest.TestCase, BaseKernelTestCase):
        def create_kernel_no_ard(self, **kwargs):
            return PeriodicKernel(**kwargs)

        def create_kernel_ard(self, num_dims, **kwargs):
            return PeriodicKernel(ard_num_dims=num_dims, **kwargs)

    class TestPeriodicKeOpsKernel(unittest.TestCase, BaseKeOpsTestCase):
        @property
        def k1(self):
            return PeriodicKernel

        @property
        def k2(self):
            return GPeriodicKernel

    def create_kernel_with_prior(self, period_length_prior):
        return self.k1(period_length_prior=period_length_prior)

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
