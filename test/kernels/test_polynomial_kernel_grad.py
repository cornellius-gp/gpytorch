#!/usr/bin/env python3

import unittest

from gpytorch.kernels import PolynomialKernelGrad
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestPolynomialKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return PolynomialKernelGrad(power=2, **kwargs)


if __name__ == "__main__":
    unittest.main()
