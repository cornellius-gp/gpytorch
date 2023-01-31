#!/usr/bin/env python3

import unittest

import torch
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator, ToeplitzLinearOperator

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestMultitaskGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 2

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        task_mat = RootLinearOperator(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        covar = KroneckerProductLinearOperator(data_mat, task_mat)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2)

    def test_marginal_variance(self):
        likelihood = MultitaskGaussianLikelihood(num_tasks=4, rank=0, has_global_noise=False)
        likelihood.task_noises = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        input = self._create_marginal_input()
        variance = likelihood(input).variance
        self.assertAllClose(variance, torch.tensor([1.1, 4.2, 9.3, 16.4]).repeat(5, 1))

        likelihood = MultitaskGaussianLikelihood(num_tasks=4, rank=1, has_global_noise=True)
        likelihood.noise = torch.tensor(0.1)
        likelihood.task_noise_covar_factor.data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        input = self._create_marginal_input()
        variance = likelihood(input).variance
        self.assertAllClose(variance, torch.tensor([2.1, 8.1, 18.1, 32.1]).repeat(5, 1))

    def test_setters(self):
        likelihood = MultitaskGaussianLikelihood(num_tasks=3, rank=0)

        a = torch.randn(3, 2)
        mat = a.matmul(a.transpose(-1, -2))

        # test rank 0 setters
        likelihood.noise = 0.5
        self.assertAlmostEqual(0.5, likelihood.noise.item())

        likelihood.task_noises = torch.tensor([0.04, 0.04, 0.04])
        for i in range(3):
            self.assertAlmostEqual(0.04, likelihood.task_noises[i].item())

        with self.assertRaises(AttributeError) as context:
            likelihood.task_noise_covar = mat
        self.assertTrue("task noises" in str(context.exception))

        # test low rank setters
        likelihood = MultitaskGaussianLikelihood(num_tasks=3, rank=2)
        likelihood.noise = 0.5
        self.assertAlmostEqual(0.5, likelihood.noise.item())

        likelihood.task_noise_covar = mat
        self.assertAllClose(mat, likelihood.task_noise_covar)

        with self.assertRaises(AttributeError) as context:
            likelihood.task_noises = torch.tensor([0.04, 0.04, 0.04])
        self.assertTrue("task noises" in str(context.exception))


class TestMultitaskGaussianLikelihoodNonInterleaved(TestMultitaskGaussianLikelihood, unittest.TestCase):
    seed = 2

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        data_mat = ToeplitzLinearOperator(torch.tensor([1, 0.6, 0.4, 0.2, 0.1]))
        task_mat = RootLinearOperator(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        covar = KroneckerProductLinearOperator(task_mat, data_mat)
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar, interleaved=False)


class TestMultitaskGaussianLikelihoodBatch(TestMultitaskGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2, batch_shape=torch.Size([3]))

    def test_nonbatch(self):
        pass
