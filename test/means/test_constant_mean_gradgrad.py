#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import ConstantMeanGradGrad
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestConstantMeanGradGrad(BaseMeanTestCase, unittest.TestCase):
    batch_shape = None

    def create_mean(self):
        return ConstantMeanGradGrad(batch_shape=self.__class__.batch_shape or torch.Size())

    def test_forward_vec(self):
        test_x = torch.randn(4)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([4, 3]))
        else:
            self.assertEqual(mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4, 3]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat(self):
        test_x = torch.randn(4, 3)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([4, 7]))
        else:
            self.assertEqual(mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4, 7]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat_batch(self):
        test_x = torch.randn(3, 4, 3)
        mean = self.create_mean()
        if self.__class__.batch_shape is None:
            self.assertEqual(mean(test_x).shape, torch.Size([3, 4, 7]))
        else:
            self.assertEqual(mean(test_x).shape, torch.Size([*self.__class__.batch_shape, 4, 7]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat_multi_batch(self):
        test_x = torch.randn(2, 3, 4, 3)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([2, 3, 4, 7]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)


class TestConstantMeanGradGradBatch(TestConstantMeanGradGrad):
    batch_shape = torch.Size([3])


class TestConstantMeanGradGradMultiBatch(TestConstantMeanGradGrad):
    batch_shape = torch.Size([2, 3])
