#!/usr/bin/env python3

import torch
import unittest
from test.means._base_mean_test_case import BaseMeanTestCase
from gpytorch.means import ConstantMean


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
