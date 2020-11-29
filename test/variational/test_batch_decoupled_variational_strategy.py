#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


def likelihood_cls():
    return gpytorch.likelihoods.GaussianLikelihood()


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return gpytorch.variational.BatchDecoupledVariationalStrategy(
        model, inducing_points, variational_distribution, learn_inducing_locations
    )


def batch_dim_strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return gpytorch.variational.BatchDecoupledVariationalStrategy(
        model, inducing_points, variational_distribution, learn_inducing_locations, mean_var_batch_dim=-1
    )


class TestBatchDecoupledVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def likelihood_cls(self):
        return likelihood_cls

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return strategy_cls

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass, and for computing prior dist
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!
        self.assertFalse(ciq_mock.called)


class TestBatchDecoupledPredictiveGP(TestBatchDecoupledVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestBatchDecoupledRobustVGP(TestBatchDecoupledVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldBatchDecoupledVariationalGP(TestBatchDecoupledVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldBatchDecoupledPredictiveGP(TestBatchDecoupledPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldBatchDecoupledRobustVGP(TestBatchDecoupledRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestBatchDecoupledVariationalGPBatchDim(TestBatchDecoupledVariationalGP, unittest.TestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.VariationalStrategy,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        class _SVGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape + torch.Size([2]))
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape + torch.Size([2])),
                    batch_shape=batch_shape + torch.Size([2]),
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        return _SVGPRegressionModel(inducing_points), self.likelihood_cls()

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestMeanFieldBatchDecoupledVariationalGPBatchDim(TestBatchDecoupledVariationalGPBatchDim, unittest.TestCase):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestBatchDecoupledVariationalGPOtherBatchDim(TestBatchDecoupledVariationalGP, unittest.TestCase):
    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.VariationalStrategy,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        class _SVGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                if constant_mean:
                    self.mean_module = gpytorch.means.ConstantMean(
                        batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:]
                    )
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:]),
                    batch_shape=batch_shape[:-1] + torch.Size([2]) + batch_shape[-1:],
                )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                return latent_pred

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        return _SVGPRegressionModel(inducing_points), self.likelihood_cls()

    @property
    def strategy_cls(self):
        def _batch_dim_strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
            return gpytorch.variational.BatchDecoupledVariationalStrategy(
                model, inducing_points, variational_distribution, learn_inducing_locations, mean_var_batch_dim=-2
            )

        return _batch_dim_strategy_cls

    @property
    def batch_shape(self):
        return torch.Size([3])


if __name__ == "__main__":
    unittest.main()
