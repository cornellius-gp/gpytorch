#!/usr/bin/env python3

import unittest

from gpytorch.kernels import ArcKernel, MaternKernel
from gpytorch.priors import NormalPrior
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestArcKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return ArcKernel(base_kernel=MaternKernel(nu=0.5), **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return ArcKernel(base_kernel=MaternKernel(nu=0.5), ard_num_dims=num_dims, **kwargs)

    def create_kernel_with_prior(self, angle_prior, radius_prior):
        return self.create_kernel_no_ard(angle_prior=angle_prior, radius_prior=radius_prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        self.create_kernel_with_prior(None, None)
        self.create_kernel_with_prior(NormalPrior(0, 1), NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, None, 1)
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1, None)
