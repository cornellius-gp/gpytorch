#!/usr/bin/env python3

import math
import unittest

import torch

from gpytorch.kernels import CylindricalKernel, MaternKernel
from gpytorch.priors import NormalPrior
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

    def create_kernel_with_prior(self, angular_weights_prior, alpha_prior, beta_prior):
        return self.create_kernel_no_ard(
            angular_weights_prior=angular_weights_prior, alpha_prior=alpha_prior, beta_prior=beta_prior
        )

    def test_prior_type(self):
        """
        Raising TypeError if prior type is other than gpytorch.priors.Prior
        """
        self.create_kernel_with_prior(None, None, None)
        self.create_kernel_with_prior(NormalPrior(0, 1), NormalPrior(0, 1), NormalPrior(0, 1))
        self.assertRaises(TypeError, self.create_kernel_with_prior, 1, None, None)
        self.assertRaises(TypeError, self.create_kernel_with_prior, None, 1, None)
        self.assertRaises(TypeError, self.create_kernel_with_prior, None, None, 1)

    def test_unit_sphere_boundary_and_origin_stability(self):
        for dtype in (torch.float, torch.double):
            kernel = CylindricalKernel(3, MaternKernel(nu=2.5)).to(dtype=dtype)
            kernel.alpha = 0.5
            raw = torch.tensor(
                [
                    [0.6, 0.8, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0],
                ],
                dtype=dtype,
            )
            # Normalize last row to unit norm (often 1 + ~2e-16 in IEEE-754)
            raw[-1] = raw[-1] / raw[-1].norm()
            x = raw.clone().requires_grad_(True)
            covar = kernel(x, x).to_dense()
            self.assertTrue(torch.isfinite(covar).all())
            covar.sum().backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())

            outside = torch.tensor([[1.05, 0.0, 0.0]], dtype=dtype)
            with self.assertRaises(RuntimeError):
                kernel(outside, outside).to_dense()


if __name__ == "__main__":
    unittest.main()
