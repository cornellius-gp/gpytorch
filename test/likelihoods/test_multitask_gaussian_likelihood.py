#!/usr/bin/env python3

import unittest

import torch

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import KroneckerProductLazyTensor, RootLazyTensor
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.test.base_likelihood_test_case import BaseLikelihoodTestCase


class TestMultitaskGaussianLikelihood(BaseLikelihoodTestCase, unittest.TestCase):
    seed = 2

    def _create_conditional_input(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def _create_marginal_input(self, batch_shape=torch.Size([])):
        mat = torch.randn(*batch_shape, 5, 5)
        mat2 = torch.randn(*batch_shape, 4, 4)
        covar = KroneckerProductLazyTensor(RootLazyTensor(mat), RootLazyTensor(mat2))
        return MultitaskMultivariateNormal(torch.randn(*batch_shape, 5, 4), covar)

    def _create_targets(self, batch_shape=torch.Size([])):
        return torch.randn(*batch_shape, 5, 4)

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2)


class TestMultitaskGaussianLikelihoodBatch(TestMultitaskGaussianLikelihood):
    seed = 0

    def create_likelihood(self):
        return MultitaskGaussianLikelihood(num_tasks=4, rank=2, batch_shape=torch.Size([3]))

    def test_nonbatch(self):
        pass
