#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


def likelihood_cls():
    return gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=4)


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    return gpytorch.variational.LMCVariationalStrategy(
        gpytorch.variational.VariationalStrategy(
            model, inducing_points, variational_distribution, learn_inducing_locations
        ),
        num_tasks=4,
        num_latents=3,
        latent_dim=-1,
    )


class TestLMCVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([3])

    @property
    def event_shape(self):
        return torch.Size([32, 4])

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
        cg_mock, _, ciq_mock = super().test_training_iteration(
            *args, expected_batch_shape=expected_batch_shape, **kwargs
        )
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        cg_mock, _, ciq_mock = super().test_eval_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)


class TestLMCPredictiveGP(TestLMCVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestLMCRobustVGP(TestLMCVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldLMCVariationalGP(TestLMCVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCPredictiveGP(TestLMCPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCRobustVGP(TestLMCRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaLMCVariationalGP(TestLMCVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCPredictiveGP(TestLMCPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCRobustVGP(TestLMCRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestLMCVariationalGPSharedVariational(TestLMCVariationalGP, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([3])


class TestLMCPredictiveGPSharedVariational(TestLMCVariationalGPSharedVariational):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestLMCRobustVGPSharedVariational(TestLMCVariationalGPSharedVariational):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldLMCVariationalGPSharedVariational(TestLMCVariationalGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCPredictiveGPSharedVariational(TestLMCPredictiveGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldLMCRobustVGPSharedVariational(TestLMCRobustVGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaLMCVariationalGPSharedVariational(TestLMCVariationalGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCPredictiveGPSharedVariational(TestLMCPredictiveGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaLMCRobustVGPSharedVariational(TestLMCRobustVGPSharedVariational):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
