#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 10:07:05 2020

@author: blue
"""
import unittest

from gpytorch.kernels import ArcKernel, MaternKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestArcKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return ArcKernel(base_kernel=MaternKernel(nu=0.5), **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return ArcKernel(base_kernel=MaternKernel(nu=0.5), ard_num_dims=num_dims, **kwargs)
