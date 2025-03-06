#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Distribution

from gpytorch.likelihoods import OrdinalLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestOrdinalLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 0

    def create_likelihood(self):
        bin_edges = torch.tensor([-0.5, 0.5])
        return OrdinalLikelihood(bin_edges)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.distributions.Categorical(probs=torch.tensor([1 / 3, 1 / 3, 1 / 3])).sample(
            torch.Size([*batch_shape, 5])
        )

    def _test_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        input = self._create_marginal_input(batch_shape)
        output = likelihood(input)

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape[-len(batch_shape) - 1 :], torch.Size([*batch_shape, 5]))
