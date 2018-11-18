#!/usr/bin/env python3

import torch
import unittest
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean


class TestMultitaskMean(unittest.TestCase):
    def setUp(self):
        self.mean = MultitaskMean([ConstantMean(), ZeroMean(), ZeroMean(), ConstantMean()], num_tasks=4)
        self.mean.base_means[0].constant.data.fill_(5)
        self.mean.base_means[3].constant.data.fill_(7)

    def test_forward(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2, 4))
        self.assertTrue(res[:, 0].eq(5).all())
        self.assertTrue(res[:, 1].eq(0).all())
        self.assertTrue(res[:, 2].eq(0).all())
        self.assertTrue(res[:, 3].eq(7).all())

    def test_forward_batch(self):
        a = torch.tensor([[[1, 2], [1, 2], [2, 4]], [[2, 3], [2, 3], [1, 3]]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2, 3, 4))
        self.assertTrue(res[:, :, 0].eq(5).all())
        self.assertTrue(res[:, :, 1].eq(0).all())
        self.assertTrue(res[:, :, 2].eq(0).all())
        self.assertTrue(res[:, :, 3].eq(7).all())


class TestMultitaskMeanSameMean(unittest.TestCase):
    def setUp(self):
        self.mean = MultitaskMean(ConstantMean(), num_tasks=4)
        self.mean.base_means[0].constant.data.fill_(0)
        self.mean.base_means[1].constant.data.fill_(1)
        self.mean.base_means[2].constant.data.fill_(2)
        self.mean.base_means[3].constant.data.fill_(3)

    def test_forward(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2, 4))
        self.assertTrue(res[:, 0].eq(0).all())
        self.assertTrue(res[:, 1].eq(1).all())
        self.assertTrue(res[:, 2].eq(2).all())
        self.assertTrue(res[:, 3].eq(3).all())

    def test_forward_batch(self):
        a = torch.tensor([[[1, 2], [1, 2], [2, 4]], [[2, 3], [2, 3], [1, 3]]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2, 3, 4))
        self.assertTrue(res[:, :, 0].eq(0).all())
        self.assertTrue(res[:, :, 1].eq(1).all())
        self.assertTrue(res[:, :, 2].eq(2).all())
        self.assertTrue(res[:, :, 3].eq(3).all())
