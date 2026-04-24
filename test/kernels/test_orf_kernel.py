#!/usr/bin/env python3

import unittest

import torch

from gpytorch.kernels import OrthogonalRandomFeaturesKernel
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase


class TestOrthogonalRandomFeaturesKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return OrthogonalRandomFeaturesKernel(num_samples=5, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        return OrthogonalRandomFeaturesKernel(num_dims=num_dims, num_samples=7, ard_num_dims=num_dims, **kwargs)

    # Override active_dims tests: the base class creates two separate kernels with
    # independent random weights, which gives different outputs for stochastic kernels.
    # We share weights explicitly, consistent with TestRFFKernel.

    def test_active_dims_list(self):
        kernel = self.create_kernel_no_ard(active_dims=[0, 2, 4, 6])
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        randn_weights = kernel.randn_weights
        kernel_basic = self.create_kernel_no_ard()
        kernel_basic._init_weights(randn_weights=randn_weights)
        covar_mat_actual = kernel_basic(x[:, [0, 2, 4, 6]]).evaluate_kernel().to_dense()
        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_active_dims_range(self):
        active_dims = list(range(3, 9))
        kernel = self.create_kernel_no_ard(active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        randn_weights = kernel.randn_weights
        kernel_basic = self.create_kernel_no_ard()
        kernel_basic._init_weights(randn_weights=randn_weights)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().to_dense()
        self.assertLess(torch.norm(covar_mat - covar_mat_actual) / covar_mat_actual.norm(), 1e-4)

    def test_weight_matrix_directions_are_orthogonal(self):
        """Within each d×d block, frequency directions (normalised rows) should be orthogonal."""
        torch.manual_seed(1)
        kernel = OrthogonalRandomFeaturesKernel(num_samples=6)
        d, D = 3, 6
        kernel._init_weights(num_dims=d, num_samples=D)
        # randn_weights shape: (d, D) -- transpose to (D, d)
        W = kernel.randn_weights.transpose(-1, -2)  # (D, d)
        for i in range(D // d):
            block = W[i * d : (i + 1) * d]  # (d, d)
            # Normalize each row to get unit directions
            unit_block = block / block.norm(dim=1, keepdim=True)
            gram = unit_block @ unit_block.T  # should be close to I_d
            self.assertLess(torch.norm(gram - torch.eye(d)).item(), 1e-4)

    def test_output_shape(self):
        """Kernel matrix should have the correct shape."""
        kernel = OrthogonalRandomFeaturesKernel(num_samples=16)
        x = torch.randn(10, 4)
        out = kernel(x, x).to_dense()
        self.assertEqual(out.shape, torch.Size([10, 10]))

    def test_output_is_symmetric(self):
        """K(x, x) should be symmetric."""
        torch.manual_seed(0)
        kernel = OrthogonalRandomFeaturesKernel(num_samples=32)
        x = torch.randn(8, 3)
        K = kernel(x, x).to_dense()
        self.assertLess(torch.norm(K - K.T).item(), 1e-5)

    def test_diagonal_is_one(self):
        """With lengthscale=1 the diagonal should equal 1 (unbiased RBF approximation)."""
        torch.manual_seed(42)
        kernel = OrthogonalRandomFeaturesKernel(num_samples=128)
        kernel.initialize(lengthscale=1.0)
        x = torch.randn(20, 4)
        K = kernel(x, x).to_dense()
        diag = K.diagonal()
        # Each z(x)^T z(x) = sum(cos^2 + sin^2) / D = 1 exactly
        self.assertLess(torch.norm(diag - 1.0).item(), 1e-5)

    def test_approximation_quality(self):
        """With enough features, ORF should closely approximate the RBF kernel."""
        torch.manual_seed(0)
        n, d, D = 15, 4, 1024
        x = torch.randn(n, d)

        kernel = OrthogonalRandomFeaturesKernel(num_samples=D)
        kernel.initialize(lengthscale=1.0)

        dists = torch.cdist(x, x).pow(2)
        K_true = torch.exp(-0.5 * dists)
        K_orf = kernel(x, x).to_dense()

        rel_error = (K_orf - K_true).norm() / K_true.norm()
        self.assertLess(rel_error.item(), 0.05, f"Relative approximation error {rel_error:.4f} > 0.05")


if __name__ == "__main__":
    unittest.main()
