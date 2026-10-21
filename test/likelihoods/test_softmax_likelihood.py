#!/usr/bin/env python3

import unittest
import warnings

import torch
from linear_operator.operators import DiagLinearOperator
from torch.distributions import Distribution

from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods import SoftmaxLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestSoftmaxLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 6)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        mat = torch.randn(*batch_shape, 6, 5, 5)
        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(torch.randn(*batch_shape, 6, 5), mat @ mat.transpose(-1, -2))
        )

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.distributions.Categorical(probs=torch.tensor([0.25, 0.25, 0.25, 0.25])).sample(
            torch.Size([*batch_shape, 5])
        )

    def create_likelihood(self):
        return SoftmaxLikelihood(num_features=6, num_classes=4)

    def _test_conditional(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_conditional_input(batch_shape)
        output = likelihood(input)

        self.assertIsInstance(output, Distribution)
        self.assertEqual(output.sample().shape, torch.Size([*batch_shape, 5]))

    def _test_log_prob(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_marginal_input(batch_shape)
        target = self._create_targets(batch_shape)
        output = likelihood.expected_log_prob(target, input)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))

    def _test_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_marginal_input(batch_shape)
        output = likelihood(input)

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape[-len(batch_shape) - 1 :], torch.Size([*batch_shape, 5]))


class TestSoftmaxLikelihoodNoMixing(TestSoftmaxLikelihood):
    seed = 0

    def create_likelihood(self):
        return SoftmaxLikelihood(num_features=6, num_classes=6, mixing_weights=False)

    def _test_learnable_parameters(self):
        likelihood = self.create_likelihood()
        self.assertEqual(len(list(likelihood.parameters())), 0)


class TestSoftmaxInputOrientation(unittest.TestCase):
    def _devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def test_modern_square_samples_values_and_derivatives(self):
        for device in self._devices():
            for mixing in (False, True):
                for batch in ((), (2,), (3, 1)):
                    with self.subTest(device=device, mixing=mixing, batch=batch):
                        likelihood = SoftmaxLikelihood(num_features=3, num_classes=3, mixing_weights=mixing).to(
                            device=device, dtype=torch.double
                        )
                        values = (
                            torch.tensor(
                                [[0.1, 0.7, -0.3], [-0.6, 0.2, 0.8], [0.4, -0.9, 0.3]],
                                device=device,
                                dtype=torch.double,
                            )
                            .expand(*batch, 3, 3)
                            .clone()
                            .requires_grad_()
                        )
                        if mixing:
                            with torch.no_grad():
                                likelihood.mixing_weights.copy_(
                                    values.new_tensor([[0.1, 0.5, 0.8], [-0.3, 0.7, 0.2], [0.9, -0.4, 0.3]])
                                )
                        logits = values @ likelihood.mixing_weights.t() if mixing else values
                        expected = torch.distributions.Categorical(logits=logits)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            actual = likelihood(values)
                        torch.testing.assert_close(actual.probs, expected.probs)
                        self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))
                        parameters = (values, *likelihood.parameters())
                        weights = values.new_tensor([0.2, -0.7, 0.4])
                        rows = []
                        for distribution in (actual, expected):
                            loss = (distribution.probs.square() * weights).sum()
                            first = torch.autograd.grad(loss, parameters, create_graph=True, retain_graph=True)
                            second = torch.autograd.grad(
                                sum(g.square().sum() for g in first), parameters, retain_graph=True
                            )
                            rows.append((first, second))
                        for observed, reference in zip(*rows):
                            for a, b in zip(observed, reference):
                                torch.testing.assert_close(a, b)

    def test_modern_square_multitask_marginal_and_expected_log_prob(self):
        class RecordingSoftmax(SoftmaxLikelihood):
            def forward(self, function_samples, *args, **kwargs):
                self.samples = function_samples
                return super().forward(function_samples, *args, **kwargs)

        for device in self._devices():
            for mixing in (False, True):
                for training in (False, True):
                    with self.subTest(device=device, mixing=mixing, training=training):
                        likelihood = (
                            RecordingSoftmax(num_features=3, num_classes=3, mixing_weights=mixing)
                            .to(device=device, dtype=torch.double)
                            .train(training)
                        )
                        mean = torch.arange(9, device=device, dtype=torch.double).reshape(3, 3) / 7
                        mean.requires_grad_()
                        covariance = DiagLinearOperator(mean.new_full((9,), 0.2))
                        distribution = MultitaskMultivariateNormal(mean, covariance)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            actual = likelihood(distribution)
                        logits = likelihood.samples
                        if mixing:
                            logits = logits @ likelihood.mixing_weights.t()
                        torch.testing.assert_close(actual.probs, logits.softmax(-1))
                        self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))
                        target = torch.tensor([0, 2, 1], device=device)
                        actual_log_prob = likelihood.expected_log_prob(target, distribution)
                        logits = likelihood.samples
                        if mixing:
                            logits = logits @ likelihood.mixing_weights.t()
                        reference = torch.distributions.Categorical(logits=logits).log_prob(target).mean(0)
                        torch.testing.assert_close(actual_log_prob, reference)

    def test_unambiguous_legacy_and_invalid_shapes(self):
        for device in self._devices():
            with self.subTest(device=device):
                likelihood = SoftmaxLikelihood(num_classes=3, mixing_weights=False).to(device)
                values = torch.arange(15, device=device, dtype=torch.float).reshape(3, 5)
                with self.assertWarns(DeprecationWarning):
                    actual = likelihood(values)
                torch.testing.assert_close(actual.probs, values.transpose(-1, -2).softmax(-1))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = likelihood(values.transpose(-1, -2))
                torch.testing.assert_close(actual.probs, values.transpose(-1, -2).softmax(-1))
                self.assertFalse(caught)
                with self.assertRaisesRegex(RuntimeError, "There should be 3 features"):
                    likelihood(torch.zeros(2, 4, device=device))
