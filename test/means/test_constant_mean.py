#!/usr/bin/env python3

import pickle
import unittest

import torch

from gpytorch.means import ConstantMean
from gpytorch.priors import NormalPrior
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestConstantMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return ConstantMean()

    def test_prior(self):
        prior = NormalPrior(0.0, 1.0)
        mean = ConstantMean(prior=prior)
        self.assertEqual(mean.mean_prior, prior)
        pickle.loads(pickle.dumps(mean))  # Should be able to pickle and unpickle with a prior
        mean._constant_closure(mean, 1.234)
        self.assertAlmostEqual(mean.constant.item(), 1.234)


class TestConstantMeanBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([3])

    def create_mean(self):
        return ConstantMean(batch_shape=self.__class__.batch_shape)


class TestConstantMeanMultiBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([2, 3])

    def create_mean(self):
        return ConstantMean(batch_shape=self.__class__.batch_shape)
