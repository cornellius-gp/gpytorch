#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.test.variational_test_case import VariationalTestCase
from gpytorch.variational.variational_strategy import ComputePredictiveUpdates


class TestVariationalGP(VariationalTestCase, unittest.TestCase):
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
        return gpytorch.variational.VariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!
        self.assertFalse(ciq_mock.called)

    def test_fantasy_call(self, *args, **kwargs):
        # we only want to check CholeskyVariationalDistribution
        if self.distribution_cls is gpytorch.variational.CholeskyVariationalDistribution:
            return super().test_fantasy_call(*args, **kwargs)

        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)

    def test_forward_train_eval(self, *args, **kwargs):
        num_inducing = 5

        model, _ = self._make_model_and_likelihood(
            num_inducing=num_inducing,
            batch_shape=self.batch_shape,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls,
        )

        train_x = torch.randn(*self.batch_shape, num_inducing + 1, 2)

        # More data than the number of inducing points; this code path uses the custom autograd function
        model.train()
        with patch.object(ComputePredictiveUpdates, "forward", wraps=ComputePredictiveUpdates.forward) as mock_forward:
            predictive_dist1 = model(train_x)
            mock_forward.assert_called()

        # This should execute the other code path without calling the custom autograd function
        model.eval()
        with patch.object(ComputePredictiveUpdates, "forward", wraps=ComputePredictiveUpdates.forward) as mock_forward:
            predictive_dist2 = model(train_x)
            mock_forward.assert_not_called()

        # The train mode and eval mode should produce the same predictive mean and variance
        self.assertAllClose(predictive_dist1.mean, predictive_dist2.mean)
        self.assertAllClose(predictive_dist1.variance, predictive_dist2.variance)


class TestPredictiveGP(TestVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestRobustVGP(TestVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldVariationalGP(TestVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldPredictiveGP(TestPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldRobustVGP(TestRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaVariationalGP(TestVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaPredictiveGP(TestPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaRobustVGP(TestRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestNGDVariationalGP(TestVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.NaturalVariationalDistribution

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = VariationalTestCase.test_training_iteration(self, *args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 6)  # Three for each forward pass
        self.assertFalse(ciq_mock.called)

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = VariationalTestCase.test_eval_iteration(self, *args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertEqual(cholesky_mock.call_count, 3)  # One to compute cache + 2 to compute variational distribution
        self.assertFalse(ciq_mock.called)


class TestComputePredictiveUpdates(BaseTestCase, unittest.TestCase):
    def create_inputs(self):
        m = 2
        n = 3

        chol = torch.rand(m, m).tril_().requires_grad_(True)
        induc_data_covar = torch.rand(m, n).requires_grad_(True)

        middle = torch.rand(m, m)
        middle = middle + middle.mT
        middle.requires_grad_(True)

        inducing_values = torch.randn(m).requires_grad_(True)

        return chol, induc_data_covar, middle, inducing_values

    def test_forward_backward(self):
        chol, induc_data_covar, middle, inducing_values = self.create_inputs()

        # Custom autograd function
        mean_update, variance_update = ComputePredictiveUpdates.apply(chol, induc_data_covar, middle, inducing_values)

        loss = mean_update.sum() + variance_update.sum()
        loss.backward()

        # Compute the ground truth
        chol_ref = chol.detach().clone().requires_grad_(True)
        induc_data_covar_ref = induc_data_covar.detach().clone().requires_grad_(True)
        middle_ref = middle.detach().clone().requires_grad_(True)
        inducing_values_ref = inducing_values.detach().clone().requires_grad_(True)

        interp_term_ref = torch.linalg.solve_triangular(chol_ref, induc_data_covar_ref, upper=False)
        mean_update_ref = (interp_term_ref.mT @ inducing_values_ref.unsqueeze(-1)).squeeze(-1)
        variance_update_ref = torch.sum(
            interp_term_ref.mT * (interp_term_ref.mT @ middle_ref),
            dim=-1,
        )

        loss_ref = mean_update_ref.sum() + variance_update_ref.sum()
        loss_ref.backward()

        # Assert that the forward outputs are the same
        self.assertAllClose(mean_update, mean_update_ref)
        self.assertAllClose(variance_update, variance_update_ref)

        # Now assert that the derivatives are the same
        self.assertAllClose(chol.grad, chol_ref.grad)
        self.assertAllClose(induc_data_covar.grad, induc_data_covar_ref.grad)
        self.assertAllClose(middle.grad, middle_ref.grad)
        self.assertAllClose(inducing_values.grad, inducing_values_ref.grad)


if __name__ == "__main__":
    unittest.main()
