#!/usr/bin/env python3

import torch
import unittest
from test.means._base_mean_test_case import BaseMeanTestCase
from gpytorch.means import HadamardMultitaskMean, ConstantMean, ZeroMean


class TestHadamardMultitaskMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return HadamardMultitaskMean([
            ConstantMean(), ZeroMean(), ZeroMean()
        ], num_tasks=3)

    def test_forward_vec(self):
        test_x = torch.randn(4)
        test_i = torch.bernoulli(torch.rand(4))
        mean = self.create_mean()
        self.assertEqual(mean(test_x, test_i).shape, torch.Size([4]))
        self.assertEqual(mean(test_x, test_i)[..., 1:].norm().item(), 0)

    def test_forward_mat(self):
        test_x = torch.randn(4, 3)
        test_i = torch.bernoulli(torch.rand(4))
        mean = self.create_mean()
        self.assertEqual(mean(test_x, test_i).shape, torch.Size([4, 3]))
        self.assertEqual(mean(test_x, test_i)[..., 1:].norm().item(), 0)

    def test_forward_mat_batch(self):
        test_x = torch.randn(3, 4, 3)
        test_i = torch.bernoulli(torch.rand(3, 3))
        test_i = torch.stack([test_i] * 4, dim=1)
        mean = self.create_mean()
        self.assertEqual(mean(test_x, test_i).shape, torch.Size([3, 4, 3]))
        self.assertEqual(mean(test_x, test_i)[..., 1:].norm().item(), 0)

    def test_forward_mat_multi_batch(self):
        test_x = torch.randn(2, 3, 4, 3)
        test_i = torch.bernoulli(torch.rand(2, 3))
        test_i = torch.stack([test_i] * 4, dim=1)
        test_i = torch.stack([test_i] * 3, dim=1)
        mean = self.create_mean()
        self.assertEqual(mean(test_x, test_i).shape, torch.Size([2, 3, 4, 3]))
        self.assertEqual(mean(test_x, test_i)[..., 1:].norm().item(), 0)


# class TestHadamardMultitaskMeanBatch(TestHadamardMultitaskMean):
#     def create_mean(self):
#         return HadamardMultitaskMean([
#             ConstantMean(batch_shape=torch.Size([3])), ZeroMean(), ZeroMean()
#         ], num_tasks=3)

#     def test_forward_vec(self):
#         test_x = torch.randn(4)
#         test_i = torch.bernoulli(torch.rand(4))
#         mean = self.create_mean()
#         self.assertEqual(mean(test_x, test_i).shape, torch.Size([4]))
#         self.assertEqual(mean(test_x, test_i)[..., 1:].norm().item(), 0)

#     def test_forward_mat(self):
#         super(TestHadamardMultitaskMeanBatch, self).test_forward_mat()


# class TestHadamardMultitaskMeanMultiBatch(TestHadamardMultitaskMean):
#     def create_mean(self):
#         return HadamardMultitaskMean([
#             ConstantMean(batch_shape=torch.Size([2, 3])), ZeroMean(), ZeroMean()
#         ], num_tasks=3)

#     def test_forward_vec(self):
#         pass

#     def test_forward_mat(self):
#         pass

#     def test_forward_mat_batch(self):
#         pass
