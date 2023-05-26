#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import LinearMeanGradGrad
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestLinearMeanGradGrad(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self, input_size=1, batch_shape=torch.Size(), bias=True, **kwargs):
        return LinearMeanGradGrad(input_size=input_size, batch_shape=batch_shape, bias=bias)

    def test_forward_vec(self):
        n = 4
        test_x = torch.randn(n)
        mean = self.create_mean(input_size=1)
        self.assertEqual(mean(test_x).shape, torch.Size([n, 3]))

    def test_forward_mat(self):
        n, d = 4, 5
        test_x = torch.randn(n, d)
        mean = self.create_mean(d)
        self.assertEqual(mean(test_x).shape, torch.Size([n, 2 * d + 1]))

    def test_forward_mat_batch(self):
        b, n, d = torch.Size([3]), 4, 5
        test_x = torch.randn(*b, n, d)
        mean = self.create_mean(d, b)
        self.assertEqual(mean(test_x).shape, torch.Size([*b, n, 2 * d + 1]))

    def test_forward_mat_multi_batch(self):
        b, n, d = torch.Size([2, 3]), 4, 5
        test_x = torch.randn(*b, n, d)
        mean = self.create_mean(d, b)
        self.assertEqual(mean(test_x).shape, torch.Size([*b, n, 2 * d + 1]))
