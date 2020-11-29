#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


def likelihood_cls():
    return gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return gpytorch.variational.IndependentMultitaskVariationalStrategy(
        gpytorch.variational.VariationalStrategy(
            model, inducing_points, variational_distribution, learn_inducing_locations
        ),
        num_tasks=2,
    )


class TestMultitaskVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([2])

    @property
    def event_shape(self):
        return torch.Size([32, 2])

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

    def test_training_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(
            *args, expected_batch_shape=expected_batch_shape, **kwargs
        )
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(
            *args, expected_batch_shape=expected_batch_shape, **kwargs
        )
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!


class TestMultitaskPredictiveGP(TestMultitaskVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestMultitaskRobustVGP(TestMultitaskVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldMultitaskVariationalGP(TestMultitaskVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldMultitaskPredictiveGP(TestMultitaskPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldMultitaskRobustVGP(TestMultitaskRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaMultitaskVariationalGP(TestMultitaskVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaMultitaskPredictiveGP(TestMultitaskPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaMultitaskRobustVGP(TestMultitaskRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
