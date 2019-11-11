#!/usr/bin/env python3
# Mostly copied from https://raw.githubusercontent.com/pyro-ppl/pyro/dev/tests/distributions/test_delta.py

import unittest

import numpy as np
import torch

import gpytorch.distributions as dist
from gpytorch.test.base_test_case import BaseTestCase


class TestDelta(BaseTestCase, unittest.TestCase):
    def setUp(self):
        self.v = torch.tensor([3.0])
        self.vs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.vs_expanded = self.vs.expand(4, 3)
        self.test_data = torch.tensor([[3.0], [3.0], [3.0]])
        self.batch_test_data_1 = torch.arange(0.0, 4.0).unsqueeze(1).expand(4, 3)
        self.batch_test_data_2 = torch.arange(4.0, 8.0).unsqueeze(1).expand(4, 3)
        self.batch_test_data_3 = torch.Tensor([[3.0], [3.0], [3.0], [3.0]])
        self.expected_support = [[[0.0], [1.0], [2.0], [3.0]]]
        self.expected_support_non_vec = [[3.0]]
        self.analytic_mean = 3.0
        self.analytic_var = 0.0
        self.n_samples = 10

    def test_log_prob_sum(self):
        log_px_torch = dist.Delta(self.v).log_prob(self.test_data).sum()
        self.assertEqual(log_px_torch.item(), 0)

    def test_batch_log_prob(self):
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_1).data
        self.assertEqual(log_px_torch.sum().item(), 0)
        log_px_torch = dist.Delta(self.vs_expanded).log_prob(self.batch_test_data_2).data
        self.assertEqual(log_px_torch.sum().item(), float("-inf"))

    def test_batch_log_prob_shape(self):
        assert dist.Delta(self.vs).log_prob(self.batch_test_data_3).size() == (4, 1)
        assert dist.Delta(self.v).log_prob(self.batch_test_data_3).size() == (4, 1)

    def test_mean_and_var(self):
        torch_samples = [dist.Delta(self.v).sample().detach().cpu().numpy() for _ in range(self.n_samples)]
        torch_mean = np.mean(torch_samples)
        torch_var = np.var(torch_samples)
        self.assertEqual(torch_mean, self.analytic_mean)
        self.assertEqual(torch_var, self.analytic_var)
