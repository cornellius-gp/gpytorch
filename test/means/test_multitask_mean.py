#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import ConstantMean, MultitaskMean, ZeroMean
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestMultitaskMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return MultitaskMean([ConstantMean(), ZeroMean(), ZeroMean()], num_tasks=3)

    def test_forward_vec(self):
        test_x = torch.randn(4)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([4, 3]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat(self):
        test_x = torch.randn(4, 3)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([4, 3]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat_batch(self):
        test_x = torch.randn(3, 4, 3)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([3, 4, 3]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)

    def test_forward_mat_multi_batch(self):
        test_x = torch.randn(2, 3, 4, 3)
        mean = self.create_mean()
        self.assertEqual(mean(test_x).shape, torch.Size([2, 3, 4, 3]))
        self.assertEqual(mean(test_x)[..., 1:].norm().item(), 0)


class TestMultitaskMeanBatch(TestMultitaskMean):
    def create_mean(self):
        return MultitaskMean([ConstantMean(batch_shape=torch.Size([3])), ZeroMean(), ZeroMean()], num_tasks=3)

    def test_forward_vec(self):
        pass

    def test_forward_mat(self):
        pass


class TestMultitaskMeanMultiBatch(TestMultitaskMean):
    def create_mean(self):
        return MultitaskMean([ConstantMean(batch_shape=torch.Size([2, 3])), ZeroMean(), ZeroMean()], num_tasks=3)

    def test_forward_vec(self):
        pass

    def test_forward_mat(self):
        pass

    def test_forward_mat_batch(self):
        pass
