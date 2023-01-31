#!/usr/bin/env python3

import unittest

import torch
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
