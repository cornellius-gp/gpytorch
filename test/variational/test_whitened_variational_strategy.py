#!/usr/bin/env python3

import unittest
import warnings
from unittest.mock import MagicMock, patch

import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.utils.warnings import ExtraComputationWarning


class TestVariationalGP(BaseTestCase, unittest.TestCase):
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
        return gpytorch.variational.WhitenedVariationalStrategy

    def _make_model_and_likelihood(
        self,
        num_inducing=16,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.VariationalStrategy,
        distribution_cls=gpytorch.variational.CholeskyVariationalDistribution,
        constant_mean=True,
    ):
        # This class is deprecated - so we'll ignore these warnings
        warnings.simplefilter("always", DeprecationWarning)

        class _SVGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points):
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, learn_inducing_locations=True
                )
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

        inducing_points = torch.randn(num_inducing, 2).repeat(*inducing_batch_shape, 1, 1)
        with warnings.catch_warnings(record=True):
            return _SVGPRegressionModel(inducing_points), self.likelihood_cls()

    def _training_iter(self, model, likelihood, batch_shape=torch.Size([]), mll_cls=gpytorch.mlls.VariationalELBO):
        # This class is deprecated - so we'll ignore these warnings
        warnings.simplefilter("always", DeprecationWarning)

        train_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        train_y = torch.linspace(-1, 1, self.event_shape[0])
        train_y = train_y.view(self.event_shape[0], *([1] * (len(self.event_shape) - 1)))
        train_y = train_y.expand(*self.event_shape)
        mll = mll_cls(likelihood, model, num_data=train_x.size(-2))

        # Single optimization iteration
        model.train()
        likelihood.train()
        with warnings.catch_warnings(record=True) as ws:
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.sum().backward()

        # Make sure we have gradients for all parameters
        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        # Make sure there were no warnings
        self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))
        return output, loss

    def _eval_iter(self, model, batch_shape=torch.Size([])):
        # This class is deprecated - so we'll ignore these warnings
        warnings.simplefilter("always", DeprecationWarning)

        test_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)

        # Single optimization iteration
        model.eval()
        with warnings.catch_warnings(record=True) as ws, torch.no_grad():
            output = model(test_x)

        # Make sure there were no warnings
        self.assertFalse(any(issubclass(w.category, ExtraComputationWarning) for w in ws))
        return output

    @property
    def event_shape(self):
        return torch.Size([32])

    @property
    def likelihood_cls(self):
        return gpytorch.likelihoods.GaussianLikelihood

    def test_eval_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        eval_data_batch_shape=None,
        expected_batch_shape=None,
    ):
        # This class is deprecated - so we'll ignore these warnings
        warnings.simplefilter("always", DeprecationWarning)

        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape
        eval_data_batch_shape = eval_data_batch_shape if eval_data_batch_shape is not None else self.batch_shape

        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.cholesky)
        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _cholesky_mock = patch("torch.cholesky", new=_wrapped_cholesky)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
        )

        # Do one forward pass
        self._training_iter(model, likelihood, data_batch_shape, mll_cls=self.mll_cls)

        # Now do evaluatioj
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock:
            # Iter 1
            _ = self._eval_iter(model, eval_data_batch_shape)
            output = self._eval_iter(model, eval_data_batch_shape)
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            return cg_mock, cholesky_mock

    def test_training_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # This class is deprecated - so we'll ignore these warnings
        warnings.simplefilter("always", DeprecationWarning)

        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        # Mocks
        _wrapped_cholesky = MagicMock(wraps=torch.cholesky)
        _wrapped_cg = MagicMock(wraps=gpytorch.utils.linear_cg)
        _cholesky_mock = patch("torch.cholesky", new=_wrapped_cholesky)
        _cg_mock = patch("gpytorch.utils.linear_cg", new=_wrapped_cg)

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
            constant_mean=constant_mean,
        )

        # Do forward pass
        with _cholesky_mock as cholesky_mock, _cg_mock as cg_mock:
            # Iter 1
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 0)
            self._training_iter(model, likelihood, data_batch_shape, mll_cls=self.mll_cls)
            self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 1)
            # Iter 2
            output, loss = self._training_iter(model, likelihood, data_batch_shape, mll_cls=self.mll_cls)
            self.assertEqual(output.batch_shape, expected_batch_shape)
            self.assertEqual(output.event_shape, self.event_shape)
            self.assertEqual(loss.shape, expected_batch_shape)
            return cg_mock, cholesky_mock
