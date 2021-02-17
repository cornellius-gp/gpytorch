#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestGridVariationalGP(VariationalTestCase, unittest.TestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=8,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.VariationalStrategy,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        class _SVGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self):
                variational_distribution = distribution_cls(num_inducing ** 2, batch_shape=batch_shape)
                variational_strategy = strategy_cls(self, num_inducing, [(-3, 3), (-3, 3)], variational_distribution)
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        return _SVGPRegressionModel(), self.likelihood_cls()

    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def learn_inducing_locations(self):
        return None

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.GridInterpolationVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        with gpytorch.settings.max_cholesky_size(0), gpytorch.settings.use_toeplitz(False):
            cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertEqual(cg_mock.call_count, 2)  # One for each forward pass
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 1)
        else:
            self.assertFalse(cholesky_mock.called)
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        with gpytorch.settings.max_cholesky_size(0):
            cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertFalse(ciq_mock.called)


class TestGridPredictiveGP(TestGridVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestGridRobustVGP(TestGridVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestGridMeanFieldVariationalGP(TestGridVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestGridMeanFieldPredictiveGP(TestGridPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestGridMeanFieldRobustVGP(TestGridRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


if __name__ == "__main__":
    unittest.main()
