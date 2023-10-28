#!/usr/bin/env python3

import unittest

import torch
from torch.nn import Parameter

from gpytorch.means import QuadraticMean

from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestQuadraticMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self, input_size, batch_shape=torch.Size()):
        return QuadraticMean(input_size=input_size, batch_shape=batch_shape)

    def test_eval(self):
        test_x = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        )
        mean = self.create_mean(input_size=3)
        A = torch.tensor(
            [
                [1.0, 0.0, -2.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        mean.A = Parameter(A)
        res = mean(test_x)
        self.assertAllClose(res, test_x.matmul(A).matmul(test_x.T).diag().div(2))

    def test_forward_vec(self):
        n = 4
        test_x = torch.randn(n)
        mean = self.create_mean(input_size=1)
        self.assertEqual(mean(test_x).shape, torch.Size([4]))

    def test_forward_mat(self):
        n, d = 4, 5
        test_x = torch.randn(n, d)
        mean = self.create_mean(d)
        self.assertEqual(mean(test_x).shape, torch.Size([n]))

    def test_forward_mat_batch(self):
        b, n, d = torch.Size([3]), 4, 5
        test_x = torch.randn(*b, n, d)
        mean = self.create_mean(d, b)
        self.assertEqual(mean(test_x).shape, torch.Size([*b, n]))

    def test_forward_mat_multi_batch(self):
        b, n, d = torch.Size([2, 3]), 4, 5
        test_x = torch.randn(*b, n, d)
        mean = self.create_mean(d, b)
        self.assertEqual(mean(test_x).shape, torch.Size([*b, n]))
