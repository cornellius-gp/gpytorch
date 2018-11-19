#!/usr/bin/env python3

import torch
import unittest
from gpytorch.means import ZeroMean


class TestZeroMean(unittest.TestCase):
    def setUp(self):
        self.mean = ZeroMean()

    def test_forward(self):
        a = torch.tensor([[1, 2], [2, 4]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2,))
        self.assertTrue(res.eq(0).all())

    def test_forward_batch(self):
        a = torch.tensor([[[1, 2], [1, 2], [2, 4]], [[2, 3], [2, 3], [1, 3]]], dtype=torch.float)
        res = self.mean(a)
        self.assertEqual(tuple(res.size()), (2, 3))
        self.assertTrue(res.eq(0).all())
