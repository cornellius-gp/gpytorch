#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.kernels import SphericalLinearKernel
from gpytorch.kernels.spherical_linear_kernel import project_onto_unit_sphere
from gpytorch.test.base_kernel_test_case import BaseKernelTestCase

UNIT_BOUNDS_3D = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
UNIT_BOUNDS_5D = torch.tensor([[0.0] * 5, [1.0] * 5])
UNIT_BOUNDS_10D = torch.tensor([[0.0] * 10, [1.0] * 10])


class TestSphericalLinearKernel(unittest.TestCase, BaseKernelTestCase):
    def create_kernel_no_ard(self, **kwargs):
        return SphericalLinearKernel(bounds=UNIT_BOUNDS_10D, **kwargs)

    def create_kernel_ard(self, num_dims, **kwargs):
        # Base class passes num_dims=2 but data is 10D; override to match data dimensionality.
        return SphericalLinearKernel(bounds=UNIT_BOUNDS_10D, ard_num_dims=10, **kwargs)

    def create_data_single_batch(self):
        return torch.randn(2, 3, 10)

    def create_data_double_batch(self):
        return torch.randn(3, 2, 50, 10)

    def test_active_dims_list(self):
        """active_dims requires bounds matching the selected dimensions."""
        active_dims = [0, 2, 4, 6]
        bounds = torch.stack([torch.zeros(len(active_dims)), torch.ones(len(active_dims))])
        kernel = SphericalLinearKernel(bounds=bounds, active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        kernel_basic = SphericalLinearKernel(bounds=bounds)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().to_dense()
        self.assertAllClose(covar_mat, covar_mat_actual, rtol=1e-3, atol=1e-5)

    def test_active_dims_range(self):
        """active_dims with a contiguous range and matching bounds."""
        active_dims = list(range(3, 9))
        bounds = torch.stack([torch.zeros(len(active_dims)), torch.ones(len(active_dims))])
        kernel = SphericalLinearKernel(bounds=bounds, active_dims=active_dims)
        x = self.create_data_no_batch()
        covar_mat = kernel(x).evaluate_kernel().to_dense()
        kernel_basic = SphericalLinearKernel(bounds=bounds)
        covar_mat_actual = kernel_basic(x[:, active_dims]).evaluate_kernel().to_dense()
        self.assertAllClose(covar_mat, covar_mat_actual, rtol=1e-3, atol=1e-5)

    def test_forward_square(self):
        """Kernel output for x1 == x2 should be a valid PSD matrix."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D)
        kernel.eval()
        x = torch.rand(10, 3)
        res = kernel(x).to_dense()
        self.assertEqual(res.shape, torch.Size([10, 10]))
        # Should be symmetric
        self.assertAllClose(res, res.T, rtol=1e-5, atol=1e-5)
        # Eigenvalues should be non-negative (PSD)
        eigvals = torch.linalg.eigvalsh(res)
        self.assertTrue((eigvals >= -1e-5).all())

    def test_forward_rectangular(self):
        """Kernel output for x1 != x2 should have correct shape."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D)
        kernel.eval()
        x1 = torch.rand(5, 3)
        x2 = torch.rand(8, 3)
        res = kernel(x1, x2).to_dense()
        self.assertEqual(res.shape, torch.Size([5, 8]))

    def test_diag(self):
        """Diagonal should always be 1 (softmax coeffs sum to 1, projection is unit norm)."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3)
        kernel.eval()
        x = torch.rand(10, 3)
        diag = kernel(x, diag=True)
        self.assertEqual(diag.shape, torch.Size([10]))
        self.assertAllClose(diag, torch.ones(10), rtol=1e-5, atol=1e-5)

    def test_custom_bounds(self):
        """Kernel should accept and use custom bounds."""
        bounds = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
        kernel = SphericalLinearKernel(bounds=bounds)
        kernel.eval()
        x = torch.rand(5, 2) * 4 - 2  # Points in [-2, 2]
        res = kernel(x).to_dense()
        self.assertEqual(res.shape, torch.Size([5, 5]))

    def test_ard(self):
        """ARD should create per-dimension lengthscales."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_5D, ard_num_dims=5)
        self.assertEqual(kernel.lengthscale.shape[-1], 5)

    def test_normalize_lengthscale(self):
        """With normalize_lengthscale, effective lengthscale should have unit L2 norm."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3, normalize_lengthscale=True)
        kernel.eval()
        x = torch.rand(5, 3)
        kernel(x).to_dense()
        effective_ls = torch.softmax(kernel.lengthscale, dim=-1).sqrt()
        self.assertAllClose(effective_ls.norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-5)

    def test_project_onto_unit_sphere(self):
        """Projected points should lie on the unit sphere."""
        x = torch.randn(10, 5)
        projected = project_onto_unit_sphere(x)
        self.assertEqual(projected.shape, torch.Size([10, 6]))
        norms = projected.norm(dim=-1)
        self.assertAllClose(norms, torch.ones(10), rtol=1e-5, atol=1e-5)

    def test_project_onto_unit_sphere_identity_on_sphere(self):
        """Unit-norm inputs in R^d map to (x, 0) on the equator of S^d."""
        d = 5
        y = torch.randn(10, d)
        y = y / y.norm(dim=-1, keepdim=True)
        projected = project_onto_unit_sphere(y)
        expected = torch.cat([y, torch.zeros(10, 1)], dim=-1)
        self.assertAllClose(projected, expected, rtol=1e-5, atol=1e-6)

    def test_gradient_flow(self):
        """Gradients should flow through the kernel computation."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3)
        x = torch.rand(5, 3)
        res = kernel(x).to_dense()
        loss = res.sum()
        loss.backward()
        for name, param in kernel.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_coeffs_setter(self):
        """Setting coeffs should round-trip through the raw parameter."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D)
        kernel.coeffs = torch.tensor([0.3, 0.7])
        self.assertAllClose(kernel.coeffs, torch.tensor([0.3, 0.7]), rtol=1e-5, atol=1e-6)

    def test_glob_ls_fraction_setter(self):
        """Setting glob_ls_fraction should round-trip through the raw parameter."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D)
        kernel.glob_ls_fraction = torch.tensor([0.4])
        self.assertAllClose(kernel.glob_ls_fraction, torch.tensor([0.4]), rtol=1e-5, atol=1e-6)

    def test_prediction_strategy(self):
        """LinearPredictionStrategy should be used for posterior inference."""
        from unittest.mock import MagicMock, patch

        train_x = torch.rand(20, 3)
        train_y = torch.randn(20)

        class _Model(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(SphericalLinearKernel(bounds=UNIT_BOUNDS_3D))

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = _Model(train_x, train_y)
        test_x = torch.rand(5, 3)

        _wrapped_ps = MagicMock(wraps=gpytorch.models.exact_prediction_strategies.LinearPredictionStrategy)
        with patch("gpytorch.models.exact_prediction_strategies.LinearPredictionStrategy", new=_wrapped_ps) as ps_mock:
            model.eval()
            output = model.likelihood(model(test_x))
            _ = output.mean + output.variance
            self.assertTrue(ps_mock.called)


if __name__ == "__main__":
    unittest.main()
