#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestUnwhitenedVariationalGP(VariationalTestCase, unittest.TestCase):
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
        return gpytorch.variational.UnwhitenedVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, once for initialization
        else:
            self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!

    def test_fantasy_call(self, *args, **kwargs):
        # we only want to check CholeskyVariationalDistribution
        if self.distribution_cls is gpytorch.variational.CholeskyVariationalDistribution:
            return super().test_fantasy_call(*args, **kwargs)

        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestUnwhitenedPredictiveGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestUnwhitenedRobustVGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestUnwhitenedMeanFieldVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedDeltaVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedBatchPrediction(unittest.TestCase):
    def test_eval_with_different_batch_sizes(self):
        inducing_points = torch.randn(20, 2)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(20)

        class GPModel(gpytorch.models.ApproximateGP):
            def __init__(self):
                variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        model = GPModel().eval()
        with torch.no_grad():
            first_output = model(torch.randn(10, 5, 2))
            second_output = model(torch.randn(2, 5, 2))

        self.assertEqual(first_output.batch_shape, torch.Size([10]))
        self.assertEqual(second_output.batch_shape, torch.Size([2]))


if __name__ == "__main__":
    unittest.main()
