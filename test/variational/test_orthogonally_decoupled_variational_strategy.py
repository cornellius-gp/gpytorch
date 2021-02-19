#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


def likelihood_cls():
    return gpytorch.likelihoods.GaussianLikelihood()


def strategy_cls(model, inducing_points, variational_distribution, learn_inducing_locations):
    base_inducing_points = torch.randn(8, inducing_points.size(-1), device=inducing_points.device)
    base_variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(8)
    return gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        gpytorch.variational.VariationalStrategy(
            model, base_inducing_points, base_variational_distribution, learn_inducing_locations
        ),
        inducing_points,
        variational_distribution,
    )


class TestOrthogonallyDecoupledVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution

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
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, and for computing prior dist

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!


class TestOrthogonallyDecoupledPredictiveGP(TestOrthogonallyDecoupledVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestOrthogonallyDecoupledRobustVGP(TestOrthogonallyDecoupledVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


if __name__ == "__main__":
    unittest.main()
