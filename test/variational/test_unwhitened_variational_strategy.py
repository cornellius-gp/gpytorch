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
        cg_mock, cholesky_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, once for initialization
        else:
            self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!


class TestUnwhitenedVariationalGP_CG(TestUnwhitenedVariationalGP):
    def test_training_iteration(self, *args, **kwargs):
        with gpytorch.settings.max_cholesky_size(0):
            cg_mock, cholesky_mock = VariationalTestCase.test_training_iteration(self, *args, **kwargs)
        self.assertEqual(cg_mock.call_count, 2)  # One for each forward pass
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 1)
        else:
            self.assertFalse(cholesky_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        with gpytorch.settings.max_cholesky_size(0):
            cg_mock, cholesky_mock = VariationalTestCase.test_eval_iteration(self, *args, **kwargs)
        self.assertEqual(cg_mock.call_count, 2)  # One for each forward pass
        self.assertFalse(cholesky_mock.called)


class TestUnwhitenedVariationalGP_CG_NoLogDet(TestUnwhitenedVariationalGP_CG):
    def test_training_iteration(self, *args, **kwargs):
        with gpytorch.settings.skip_logdet_forward(True):
            super().test_training_iteration(*args, **kwargs)

    def test_eval_iteration(self, *args, **kwargs):
        with gpytorch.settings.skip_logdet_forward(True):
            super().test_eval_iteration(*args, **kwargs)


class TestUnwhitenedVariationalGP_CG_NoPosteriorVariance(TestUnwhitenedVariationalGP_CG):
    def test_training_iteration(self, *args, **kwargs):
        with gpytorch.settings.skip_posterior_variances(True):
            super().test_training_iteration(*args, **kwargs)

    def test_eval_iteration(self, *args, **kwargs):
        with gpytorch.settings.skip_posterior_variances(True), gpytorch.settings.max_cholesky_size(0):
            cg_mock, cholesky_mock = VariationalTestCase.test_eval_iteration(self, *args, **kwargs)
        self.assertEqual(cg_mock.call_count, 1)  # One for the cache - and that's it!
        self.assertFalse(cholesky_mock.called)


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


if __name__ == "__main__":
    unittest.main()
