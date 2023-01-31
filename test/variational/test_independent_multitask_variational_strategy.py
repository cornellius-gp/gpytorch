#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


def multitask_likelihood_cls():
    return gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)


def singletask_likelihood_cls():
    return gpytorch.likelihoods.GaussianLikelihood()


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
        return multitask_likelihood_cls

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return strategy_cls

    def test_training_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        super().test_training_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)

    def test_eval_iteration(self, *args, expected_batch_shape=None, **kwargs):
        expected_batch_shape = expected_batch_shape or self.batch_shape
        expected_batch_shape = expected_batch_shape[:-1]
        super().test_eval_iteration(*args, expected_batch_shape=expected_batch_shape, **kwargs)

    def test_fantasy_call(self, *args, **kwargs):
        with self.assertRaises(AttributeError):
            super().test_fantasy_call(*args, **kwargs)


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


class TestIndexedMultitaskVariationalGP(TestMultitaskVariationalGP, unittest.TestCase):
    def _training_iter(
        self, model, likelihood, batch_shape=torch.Size([]), mll_cls=gpytorch.mlls.VariationalELBO, cuda=False
    ):
        batch_shape = list(batch_shape)
        batch_shape[-1] = 1
        train_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        train_i = torch.rand(*batch_shape, 32).round().long()
        train_y = torch.linspace(-1, 1, self.event_shape[0])
        train_y = train_y.view(self.event_shape[0], *([1] * (len(self.event_shape) - 1)))
        train_y = train_y.expand(*self.event_shape)
        mll = mll_cls(likelihood, model, num_data=train_x.size(-2))
        if cuda:
            train_x = train_x.cuda()
            train_i = train_i.cuda()
            train_y = train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Single optimization iteration
        model.train()
        likelihood.train()
        output = model(train_x, task_indices=train_i)
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

    def _eval_iter(self, model, batch_shape=torch.Size([]), cuda=False):
        batch_shape = list(batch_shape)
        batch_shape[-1] = 1
        test_x = torch.randn(*batch_shape, 32, 2).clamp(-2.5, 2.5)
        test_i = torch.rand(*batch_shape, 32).round().long()
        if cuda:
            test_x = test_x.cuda()
            test_i = test_i.cuda()
            model = model.cuda()

        # Single optimization iteration
        model.eval()
        with torch.no_grad():
            output = model(test_x, task_indices=test_i)

        return output

    @property
    def event_shape(self):
        return torch.Size([32])

    @property
    def likelihood_cls(self):
        return singletask_likelihood_cls


class TestIndexedMultitaskPredictiveGP(TestIndexedMultitaskVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestIndexedMultitaskRobustVGP(TestIndexedMultitaskVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestMeanFieldIndexedMultitaskVariationalGP(TestIndexedMultitaskVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedMultitaskPredictiveGP(TestIndexedMultitaskPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestMeanFieldIndexedMultitaskRobustVGP(TestIndexedMultitaskRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaIndexedMultitaskVariationalGP(TestIndexedMultitaskVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedMultitaskPredictiveGP(TestIndexedMultitaskPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestDeltaIndexedMultitaskRobustVGP(TestIndexedMultitaskRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
