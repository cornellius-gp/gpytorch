#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestVNNGPNonInducingData(VariationalTestCase, unittest.TestCase):
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

    # VNNGP specific
    @property
    def full_batch(self):
        return False

    @property
    def computed_full_kl(self):
        return False

    def _make_model_and_likelihood(
        self,
        num_inducing=32,
        batch_shape=torch.Size([]),
        inducing_batch_shape=torch.Size([]),
        strategy_cls=gpytorch.variational.NNVariationalStrategy,
        distribution_cls=gpytorch.variational.MeanFieldVariationalDistribution,
        constant_mean=True,
    ):
        # VNNGP variational strategy takes slightly different inputs than other variational strategies
        # (i.e. it does not accept a learn_inducing_locations argument, and it expects
        # a k and training_batch_size argument)
        # We supply a custom method here for that purpose

        class _VNNGPRegressionModel(gpytorch.models.ApproximateGP):
            def __init__(self, inducing_points, k, training_batch_size, compute_full_kl):
                variational_distribution = distribution_cls(num_inducing, batch_shape=batch_shape)
                variational_strategy = strategy_cls(
                    self,
                    inducing_points,
                    variational_distribution,
                    k=k,
                    training_batch_size=training_batch_size,
                    compute_full_kl=compute_full_kl,
                )
                super().__init__(variational_strategy)

                if constant_mean:
                    self.mean_module = gpytorch.means.ConstantMean()
                    self.mean_module.initialize(constant=1.0)
                else:
                    self.mean_module = gpytorch.means.ZeroMean()

                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=2),
                    batch_shape=batch_shape,
                )

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
        training_batch_size = num_inducing if self.full_batch else 4
        compute_full_kl = self.computed_full_kl
        inducing_points = torch.randn(*inducing_batch_shape, num_inducing, d)
        return _VNNGPRegressionModel(inducing_points, k, training_batch_size, compute_full_kl), self.likelihood_cls()

    def test_training_iteration_batch_data(self):
        # Data batch shape must always be subsumed by the inducing batch shape for VNNGP models
        # So this test does not apply to VNNGP models
        pass

    def test_eval_smaller_pred_batch(self):
        # Data batch shape must always be subsumed by the inducing batch shape for VNNGP models
        # So this test does not apply to VNNGP models
        pass

    def test_eval_larger_pred_batch(self):
        # Data batch shape must always be subsumed by the inducing batch shape for VNNGP models
        # So this test does not apply to VNNGP models
        pass

    def test_training_all_batch_zero_mean(self):
        # Original test in VariationalTestCase has a data_batch_shape that is not subsumed
        # by the inducing_batch_shape (not allowed for VNNGP models).
        return self.test_training_iteration(
            model_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            inducing_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            data_batch_shape=(torch.Size([4]) + self.batch_shape),
            expected_batch_shape=(torch.Size([3, 4]) + self.batch_shape),
            constant_mean=False,
        )

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)


class TestVNNGP(TestVNNGPNonInducingData, unittest.TestCase):
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
        output = model(x=None)
        current_training_indices = model.variational_strategy.current_training_indices
        y_batch = train_y[..., current_training_indices]
        loss = -mll(output, y_batch)
        loss.sum().backward()

        # Make sure we have gradients for all parameters
        for _, param in model.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)
        for _, param in likelihood.named_parameters():
            self.assertTrue(param.grad is not None)
            self.assertGreater(param.grad.norm().item(), 0)

        return output, loss

    def test_training_iteration(
        self,
        data_batch_shape=None,
        inducing_batch_shape=None,
        model_batch_shape=None,
        expected_batch_shape=None,
        constant_mean=True,
    ):
        # We cannot inheret the superclass method because it expects the
        # expected output.event_shape should be the training_batch_size not
        # self.event_shape (which is reserved for test_eval_iteration)

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
        self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 0)
        self._training_iter(
            model,
            likelihood,
            data_batch_shape,
            mll_cls=self.mll_cls,
            cuda=self.cuda,
        )
        # Iter 2
        self.assertEqual(model.variational_strategy.variational_params_initialized.item(), 1)
        output, loss = self._training_iter(
            model,
            likelihood,
            data_batch_shape,
            mll_cls=self.mll_cls,
            cuda=self.cuda,
        )
        self.assertEqual(output.batch_shape, expected_batch_shape)
        self.assertEqual(output.event_shape, torch.Size([model.variational_strategy.training_batch_size]))
        self.assertEqual(loss.shape, expected_batch_shape)


class TestVNNGPFullBatch(TestVNNGP, unittest.TestCase):
    @property
    def full_batch(self):
        return True


class TestVNNGPFullKL(TestVNNGP, unittest.TestCase):
    @property
    def compute_full_kl(self):
        return True


if __name__ == "__main__":
    unittest.main()
