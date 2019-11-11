#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import ConstantMean
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestConstantMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return ConstantMean()


class TestConstantMeanBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([3])

    def create_mean(self):
        return ConstantMean(batch_shape=self.__class__.batch_shape)


class TestConstantMeanMultiBatch(BaseMeanTestCase, unittest.TestCase):
    batch_shape = torch.Size([2, 3])

    def create_mean(self):
        return ConstantMean(batch_shape=self.__class__.batch_shape)
