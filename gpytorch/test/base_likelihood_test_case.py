#!/usr/bin/env python3

from abc import abstractmethod

import torch
from torch.distributions import Distribution

import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood

from .base_test_case import BaseTestCase


class BaseLikelihoodTestCase(BaseTestCase):
    @abstractmethod
    def create_likelihood(self, **kwargs):
        raise NotImplementedError()

    def _create_conditional_input(self, batch_shape=torch.Size()):
        return torch.randn(*batch_shape, 5)

    def _create_marginal_input(self, batch_shape=torch.Size()):
        mat = torch.randn(*batch_shape, 5, 5)
        eye = torch.diag_embed(torch.ones(*batch_shape, 5))
        return MultivariateNormal(torch.randn(*batch_shape, 5), mat @ mat.transpose(-1, -2) + eye)

    def _create_targets(self, batch_shape=torch.Size()):
        return torch.randn(*batch_shape, 5)

    def _test_conditional(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_conditional_input(batch_shape)
        output = likelihood(input)

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape, input.shape)

    def _test_log_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        target = self._create_targets(batch_shape)
        with gpytorch.settings.num_likelihood_samples(512):
            output = likelihood.log_marginal(target, input)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))
        with gpytorch.settings.num_likelihood_samples(512):
            default_log_prob = Likelihood.log_marginal(likelihood, target, input)
        self.assertAllClose(output, default_log_prob, rtol=0.25)

    def _test_log_prob(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        target = self._create_targets(batch_shape)
        with gpytorch.settings.num_likelihood_samples(512):
            output = likelihood.expected_log_prob(target, input)

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, batch_shape + torch.Size([5]))
        with gpytorch.settings.num_likelihood_samples(512):
            default_log_prob = Likelihood.expected_log_prob(likelihood, target, input)
        self.assertAllClose(output, default_log_prob, rtol=0.25)

    def _test_marginal(self, batch_shape):
        likelihood = self.create_likelihood()
        likelihood.max_plate_nesting += len(batch_shape)
        input = self._create_marginal_input(batch_shape)
        output = likelihood(input)

        self.assertTrue(isinstance(output, Distribution))
        self.assertEqual(output.sample().shape[-len(input.sample().shape) :], input.sample().shape)

        # Compare against default implementation
        with gpytorch.settings.num_likelihood_samples(30000):
            default = Likelihood.marginal(likelihood, input)
        # print(output.mean, default.mean)
        default_mean = default.mean
        actual_mean = output.mean
        if default_mean.dim() > actual_mean.dim():
            default_mean = default_mean.mean(0)
        self.assertAllClose(default_mean, actual_mean, rtol=0.25, atol=0.25)

    def test_nonbatch(self):
        self._test_conditional(batch_shape=torch.Size([]))
        self._test_log_marginal(batch_shape=torch.Size([]))
        self._test_log_prob(batch_shape=torch.Size([]))
        self._test_marginal(batch_shape=torch.Size([]))

    def test_batch(self):
        self._test_conditional(batch_shape=torch.Size([3]))
        self._test_log_marginal(batch_shape=torch.Size([3]))
        self._test_log_prob(batch_shape=torch.Size([3]))
        self._test_marginal(batch_shape=torch.Size([3]))

    def test_multi_batch(self):
        self._test_conditional(batch_shape=torch.Size([2, 3]))
        self._test_log_marginal(batch_shape=torch.Size([2, 3]))
        self._test_log_prob(batch_shape=torch.Size([2, 3]))
        self._test_marginal(batch_shape=torch.Size([2, 3]))
