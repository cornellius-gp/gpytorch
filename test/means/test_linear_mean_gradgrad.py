#!/usr/bin/env python3

import unittest

import torch

from gpytorch.means import LinearMean, LinearMeanGradGrad
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

    def test_broadcast_parameter_batches(self):
        n, d = 4, 2
        for input_batch, parameter_batch in (((), (3,)), ((1,), (3,)), ((2, 1), (1, 3)), ((2, 3), (3,)), ((), ())):
            for bias in (False, True):
                with self.subTest(input_batch=input_batch, parameter_batch=parameter_batch, bias=bias):
                    x = torch.randn(*input_batch, n, d, dtype=torch.double)
                    mean = self.create_mean(d, torch.Size(parameter_batch), bias=bias).double()
                    reference = LinearMean(d, batch_shape=torch.Size(parameter_batch), bias=bias).double()
                    reference.load_state_dict(mean.state_dict())
                    batch = torch.broadcast_shapes(input_batch, parameter_batch)
                    expanded_x = x.expand(*batch, n, d).detach().requires_grad_(True)
                    values = reference(expanded_x)
                    derivatives = torch.autograd.grad(values.sum(), expanded_x)[0]

                    actual = mean(x)
                    torch.testing.assert_close(actual[..., 0], values)
                    torch.testing.assert_close(actual[..., 1 : d + 1], derivatives)
                    torch.testing.assert_close(actual[..., d + 1 :], torch.zeros_like(derivatives))
