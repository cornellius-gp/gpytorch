#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import CylindricalKernel, MaternKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestCylindricalKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return CylindricalKernel(5, MaternKernel(nu=2.5), **kwargs)

    def create_data_no_batch(self):
        return torch.rand(50, 10) / math.sqrt(10)

    def create_data_single_batch(self):
        return torch.rand(2, 50, 2) / math.sqrt(2)

    def create_data_double_batch(self):
        return torch.rand(3, 2, 50, 2) / math.sqrt(2)


if __name__ == "__main__":
    unittest.main()
