#!/usr/bin/env python3

import pickle
import unittest

import torch

from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestConstantMean(BaseMeanTestCase, unittest.TestCase):
    batch_shape = None

    def create_mean(self, prior=None):
        return ConstantMean(prior=prior, batch_shape=torch.Size([]))

    def test_prior(self):
        if self.batch_shape is None:
            prior = NormalPrior(0.0, 1.0)
        else:
            prior = NormalPrior(torch.zeros(self.batch_shape), torch.ones(self.batch_shape))
        mean = self.create_mean(prior=prior)
        self.assertEqual(mean.mean_prior, prior)
        pickle.loads(pickle.dumps(mean))  # Should be able to pickle and unpickle with a prior
        value = prior.sample()
        mean._constant_closure(mean, value)
        self.assertTrue(torch.equal(mean.constant.data, value.reshape(mean.constant.data.shape)))


class TestConstantMeanBatch(TestConstantMean, unittest.TestCase):
    batch_shape = torch.Size([3])

    def create_mean(self, prior=None):
        return ConstantMean(prior=prior, batch_shape=self.__class__.batch_shape)


class TestConstantMeanMultiBatch(TestConstantMean, unittest.TestCase):
    batch_shape = torch.Size([2, 3])

    def create_mean(self, prior=None):
        return ConstantMean(prior=prior, batch_shape=self.__class__.batch_shape)
