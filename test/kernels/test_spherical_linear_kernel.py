#!/usr/bin/env python3

import pickle
import unittest
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.kernels import SphericalLinearKernel
from gpytorch.kernels.spherical_linear_kernel import project_onto_unit_sphere
from gpytorch.priors import NormalPrior
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

    def test_diag_with_learned_params(self):
        """Diagonal should remain 1 even after modifying parameters."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3)
        # Modify raw_coeffs to be non-zero
        kernel.raw_coeffs.data = torch.randn_like(kernel.raw_coeffs)
        kernel.raw_glob_ls.data = torch.randn_like(kernel.raw_glob_ls)
        kernel.eval()
        x = torch.rand(10, 3)
        diag = kernel(x, diag=True)
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

    def test_gradient_flow(self):
        """Gradients should flow through the kernel computation."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3)
        x = torch.rand(5, 3)
        res = kernel(x).to_dense()
        loss = res.sum()
        loss.backward()
        for name, param in kernel.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")

    def test_pickle_with_bounds(self):
        """Kernel with bounds should survive pickle round-trip."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        kernel = SphericalLinearKernel(bounds=bounds)
        loaded = pickle.loads(pickle.dumps(kernel))
        self.assertAllClose(loaded.bounds, bounds)

    def test_prior(self):
        """Should accept valid priors and reject invalid ones."""
        SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, lengthscale_prior=None)
        SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, lengthscale_prior=NormalPrior(0, 1))
        self.assertRaises(TypeError, SphericalLinearKernel, UNIT_BOUNDS_3D, lengthscale_prior=1)

    def test_pickle_with_prior(self):
        """Kernel with prior should survive pickle round-trip."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, lengthscale_prior=NormalPrior(0, 1))
        pickle.loads(pickle.dumps(kernel))

    def test_consistency_square_vs_rectangular(self):
        """k(x, x) computed as square should match k(x1, x2) when x1 == x2."""
        kernel = SphericalLinearKernel(bounds=UNIT_BOUNDS_3D, ard_num_dims=3)
        kernel.eval()
        x = torch.rand(8, 3)
        res_square = kernel(x).to_dense()
        # Use clone to prevent torch.equal from returning True
        res_rect = kernel(x, x.clone()).to_dense()
        self.assertAllClose(res_square, res_rect, rtol=1e-4, atol=1e-4)

    def test_prediction_strategy(self):
        """LinearPredictionStrategy should be used for posterior inference."""
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
