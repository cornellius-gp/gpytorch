#!/usr/bin/env python3

import unittest

import torch

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

    def test_empty_inputs(self):
        for batch_shape in (torch.Size(), torch.Size([2])):
            kernel = IndexKernel(num_tasks=2, batch_shape=batch_shape).double()
            for n, m in ((0, 0), (0, 3), (3, 0)):
                with self.subTest(batch_shape=batch_shape, n=n, m=m):
                    x = torch.zeros(n, 1, dtype=torch.long)
                    y = torch.zeros(m, 1, dtype=torch.long)
                    covariance = kernel(x, y).to_dense()
                    self.assertEqual(covariance.shape, batch_shape + torch.Size([n, m]))
                    self.assertEqual(covariance.dtype, kernel.covar_factor.dtype)
                    self.assertEqual(covariance.device, kernel.covar_factor.device)
