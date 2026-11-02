#!/usr/bin/env python3

import math
import unittest

import torch
from linear_operator.operators import DiagLinearOperator

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.variational_test_case import VariationalTestCase


class TestCiqVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.CiqVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 4)  # One for each forward pass, one for each backward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 2)  # One for each evaluation call

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestMeanFieldCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestNgdCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.NaturalVariationalDistribution


class TestNgdCiqFullCovariance(BaseTestCase, unittest.TestCase):
    r"""Regression tests for the predictive covariance of :class:`~gpytorch.variational.CiqVariationalStrategy`.

    The natural gradient code path used to only compute the marginal predictive variances, and therefore always
    returned a ``DiagLinearOperator``. That silently dropped every off-diagonal entry of the predictive covariance of
    ``model(x, diag=False)``, as well as all predictive covariances in eval mode (where the full covariance is required
    regardless of ``diag``).
    """

    seed = 0
    num_inducing = 6
    num_data = 5
    lengthscale = 0.25

    def setUp(self):
        super().setUp()
        self.orig_dtype = torch.get_default_dtype()
        # float64 makes the CIQ / CG solves accurate enough to compare against the standard whitened strategy
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.orig_dtype)
        super().tearDown()

    def _make_inputs(self, batch_shape):
        # the data points are interleaved with the inducing points, so that the posterior correlations are substantial
        inducing_points = torch.linspace(-1.0, 1.0, self.num_inducing).unsqueeze(-1)
        test_x = torch.linspace(-0.9, 0.9, self.num_data).unsqueeze(-1)
        if len(batch_shape):
            inducing_points = inducing_points.expand(*batch_shape, -1, -1).clone()
            test_x = test_x.expand(*batch_shape, -1, -1).clone()
        return inducing_points, test_x

    def _make_natural_params(self, batch_shape):
        root = torch.randn(*batch_shape, self.num_inducing, self.num_inducing)
        precision = root @ root.transpose(-1, -2) / self.num_inducing + 2.0 * torch.eye(self.num_inducing)
        return 0.4 * torch.randn(*batch_shape, self.num_inducing), precision.mul(-0.5)

    def _make_model(self, strategy_cls, batch_shape, inducing_points, natural_vec, natural_mat):
        num_inducing = inducing_points.size(-2)

        class _SVGPModel(gpytorch.models.ApproximateGP):
            def __init__(self):
                variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
                    num_inducing_points=num_inducing, batch_shape=batch_shape
                )
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=False
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
                self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=batch_shape)

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = _SVGPModel()
        model.covar_module.raw_lengthscale.data.fill_(math.log(math.expm1(self.lengthscale)))
        variational_distribution = model.variational_strategy._variational_distribution
        variational_distribution.natural_vec.data.copy_(natural_vec)
        variational_distribution.natural_mat.data.copy_(natural_mat)
        model.variational_strategy.variational_params_initialized.fill_(1)
        return model

    def _check_full_covariance(self, batch_shape):
        inducing_points, test_x = self._make_inputs(batch_shape)
        # Prior parameters: natural_vec = 0, natural_mat = -0.5 * I -> q(u) = p(u)
        natural_vec = torch.zeros(*batch_shape, self.num_inducing)
        natural_mat = torch.eye(self.num_inducing).mul(-0.5).expand(*batch_shape, -1, -1)

        ciq_model = self._make_model(
            gpytorch.variational.CiqVariationalStrategy, batch_shape, inducing_points, natural_vec, natural_mat
        )
        standard_model = self._make_model(
            gpytorch.variational.VariationalStrategy, batch_shape, inducing_points, natural_vec, natural_mat
        )

        standard_model.eval()
        with torch.no_grad():
            expected = standard_model(test_x, diag=False)
        expected_mean = expected.mean
        expected_covar = expected.lazy_covariance_matrix.to_dense()

        # In eval mode, the full covariance has to be returned even when diag is True (its default)
        ciq_model.eval()
        with torch.no_grad():
            for output in (ciq_model(test_x, diag=False), ciq_model(test_x)):
                self.assertAllClose(output.mean, expected_mean, rtol=1e-6, atol=1e-6)
                self.assertAllClose(output.lazy_covariance_matrix.to_dense(), expected_covar, rtol=1e-5, atol=1e-6)

        ciq_model.train()
        with torch.no_grad():
            full_output = ciq_model(test_x, diag=False)
            self.assertAllClose(full_output.lazy_covariance_matrix.to_dense(), expected_covar, rtol=1e-5, atol=1e-6)

            # The O(nM) marginal path is still used for the fast training-mode predictions
            marginal_output = ciq_model(test_x, diag=True)
            self.assertIsInstance(marginal_output.lazy_covariance_matrix, DiagLinearOperator)
            self.assertAllClose(
                marginal_output.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2),
                expected_covar.diagonal(dim1=-1, dim2=-2),
                rtol=1e-5,
                atol=1e-6,
            )

        # With q(u) = p(u), the predictive covariance has to be the prior covariance K_XX
        prior_covar = ciq_model.covar_module(test_x).to_dense()
        off_diagonal = ~torch.eye(self.num_data, dtype=torch.bool)
        self.assertAllClose(expected_covar[..., off_diagonal], prior_covar[..., off_diagonal], rtol=1e-5, atol=1e-6)
        self.assertGreater(expected_covar[..., off_diagonal].abs().max().item(), 0.1)

    def test_full_covariance(self):
        for batch_shape in (torch.Size([]), torch.Size([2])):
            self._check_full_covariance(batch_shape)

    def test_full_covariance_gradients_symmetric(self):
        r"""Full covariance gradients must be symmetric and nonzero for both symmetric and nonsymmetric loss weights."""
        batch_shape = torch.Size([])
        inducing_points, test_x = self._make_inputs(batch_shape)
        natural_vec, natural_mat = self._make_natural_params(batch_shape)

        ciq_model = self._make_model(
            gpytorch.variational.CiqVariationalStrategy, batch_shape, inducing_points, natural_vec, natural_mat
        )

        # Test with deliberately nonsymmetric weights to ensure the gradient w.r.t. natural_mat is symmetrized
        weights = torch.randn(self.num_data, self.num_data, dtype=torch.float64)
        output = ciq_model(test_x, diag=False)
        loss = (output.lazy_covariance_matrix.to_dense() * weights).sum()
        loss.backward()

        dist = ciq_model.variational_strategy._variational_distribution
        self.assertIsNotNone(dist.natural_vec.grad)
        self.assertIsNotNone(dist.natural_mat.grad)
        self.assertGreater(dist.natural_vec.grad.norm().item(), 1e-4)
        self.assertGreater(dist.natural_mat.grad.norm().item(), 1e-4)

        # The natural_mat gradient MUST be symmetric even with nonsymmetric loss weights
        mat_grad = dist.natural_mat.grad
        self.assertAllClose(mat_grad, mat_grad.transpose(-1, -2), rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
