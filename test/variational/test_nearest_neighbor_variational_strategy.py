#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestVNNGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.NNVariationalStrategy

    @property
    def likelihood_cls(self):
        return gpytorch.likelihoods.GaussianLikelihood

    @property
    def event_shape(self):
        return torch.Size([32])

    def _make_model_and_likelihood(
        self,
        num_inducing=32,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.NNVariationalStrategy,
        distribution_cls=gpytorch.variational.MeanFieldVariationalDistribution,
        constant_mean=True,
    ):
        class _VNNGPRegressionModel(gpytorch.models.GP):
            def __init__(self, inducing_points, k, training_batch_size):
                super(_VNNGPRegressionModel, self).__init__()

                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape)

                self.variational_strategy = strategy_cls(
                    self, inducing_points, variational_distribution, k=k, training_batch_size=training_batch_size
                )

                if constant_mean:
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = gpytorch.means.ZeroMean()

                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            def __call__(self, x, prior=False, **kwargs):
                if x is not None:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                return self.variational_strategy(x=x, prior=False, **kwargs)

        k = 3
        d = 2
        training_batch_size = 4
        inducing_points = torch.randn(*inducing_batch_shape, num_inducing, d)
        return _VNNGPRegressionModel(inducing_points, k, training_batch_size), self.likelihood_cls()

    def _training_iter(
        self,
        model,
        likelihood,
        batch_shape=torch.Size([]),
        mll_cls=gpytorch.mlls.VariationalELBO,
        cuda=False,
    ):
        # We cannot inheret the superclass method
        # Because it sets the training data to be the inducing points

        train_x = model.variational_strategy.inducing_points
        train_y = torch.randn(train_x.shape[:-1])
        mll = mll_cls(likelihood, model, num_data=train_x.size(-2))
        if cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Single optimization iteration
        model.train()
        likelihood.train()
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

        return output, loss

    def _eval_iter(self, model, cuda=False):
        inducing_batch_shape = model.variational_strategy.inducing_points.shape[:-2]
        test_x = torch.randn(*inducing_batch_shape, 32, 2).clamp(-2.5, 2.5)
        if cuda:
            test_x = test_x.cuda()
            model = model.cuda()

        # Single optimization iteration
        model.eval()
        with torch.no_grad():
            output = model(test_x)

        return output

    def test_training_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # We cannot inheret the superclass method
        # Because it expects `variational_params_intialized` to be set to 0

        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        data_batch_shape = data_batch_shape if data_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
            distribution_cls=self.distribution_cls,
            strategy_cls=self.strategy_cls,
            constant_mean=constant_mean,
        )

        # Do forward pass
        # Iter 1
        self._training_iter(
            model,
            likelihood,
            data_batch_shape,
            mll_cls=self.mll_cls,
            cuda=self.cuda,
        )
        # Iter 2
        output, loss = self._training_iter(
            model,
            likelihood,
            data_batch_shape,
            mll_cls=self.mll_cls,
            cuda=self.cuda,
        )
        self.assertEqual(output.batch_shape, expected_batch_shape)
        self.assertEqual(output.event_shape, self.event_shape)
        self.assertEqual(loss.shape, expected_batch_shape)

    def test_training_iteration_batch_inducing(self):
        # We need different batch sizes than the superclass
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_iteration_batch_data(self):
        # We need different batch sizes than the superclass
        return self.test_training_iteration(
            model_batch_shape=self.batch_shape,
            inducing_batch_shape=self.batch_shape,
            expected_batch_shape=(self.batch_shape),
        )

    def test_training_iteration_batch_model(self):
        # We need different batch sizes than the superclass
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3]) + self.batch_shape),
            inducing_batch_shape=self.batch_shape,
            expected_batch_shape=(torch.Size([3]) + self.batch_shape),
        )

    def test_training_all_batch_zero_mean(self):
        # We need different batch sizes than the superclass
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            constant_mean=False,
        )

    def test_eval_iteration(
        self,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
    ):
        # Batch shapes
        model_batch_shape = model_batch_shape if model_batch_shape is not None else self.batch_shape
        inducing_batch_shape = inducing_batch_shape if inducing_batch_shape is not None else self.batch_shape
        expected_batch_shape = expected_batch_shape if expected_batch_shape is not None else self.batch_shape

        # Make model and likelihood
        model, likelihood = self._make_model_and_likelihood(
            batch_shape=model_batch_shape,
            inducing_batch_shape=inducing_batch_shape,
        )

        # Do one forward pass
        self._training_iter(model, likelihood, mll_cls=self.mll_cls, cuda=self.cuda)

        # Now do evaluation
        # Iter 1
        _ = self._eval_iter(model, cuda=self.cuda)
        output = self._eval_iter(model, cuda=self.cuda)
        self.assertEqual(output.batch_shape, expected_batch_shape)
        self.assertEqual(output.event_shape, self.event_shape)

    def test_eval_smaller_pred_batch(self):
        # We need different batch sizes than the superclass
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 1]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
        )

    def test_eval_larger_pred_batch(self):
        # We need different batch sizes than the superclass
        return self.test_eval_iteration(
            model_batch_shape=(torch.Size([4]) + self.batch_shape),
            inducing_batch_shape=(self.batch_shape),
            expected_batch_shape=(torch.Size([4]) + self.batch_shape),
        )

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(AttributeError):
            super().test_fantasy_call(*args, **kwargs)


if __name__ == "__main__":
    unittest.main()
