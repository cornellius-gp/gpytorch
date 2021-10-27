#!/usr/bin/env python3

import unittest

from gpytorch.kernels import IndexKernel
from gpytorch.priors import NormalPrior


class TestIndexKernel(unittest.TestCase):
    def create_kernel_with_prior(self, prior):
        return IndexKernel(num_tasks=1, prior=prior)

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        kernel_fn = lambda prior: self.create_kernel_with_prior(prior)
        kernel_fn(None)
        kernel_fn(NormalPrior(0, 1))
        self.assertRaises(TypeError, kernel_fn, 1)
