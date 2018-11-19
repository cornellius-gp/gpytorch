#!/usr/bin/env python3

import unittest

import os
import random
import math
import torch
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.lazy import DiagLazyTensor
from test._utils import approx_equal


class TestMultiTaskMultivariateNormal(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(1)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(1)
            random.seed(1)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_multitask_multivariate_normal_exceptions(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1], dtype=torch.float, device=device)
        covmat = torch.eye(2, device=device)
        with self.assertRaises(RuntimeError):
            MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

    def test_multitask_multivariate_normal_exceptions_cuda(self):
        if torch.cuda.is_available():
            self.test_multitask_multivariate_normal_exceptions(cuda=True)

    def test_multitask_multivariate_normal(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float, device=device)
        variance = 1 + torch.arange(4, dtype=torch.float, device=device)
        covmat = torch.diag(variance)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
        self.assertTrue(torch.equal(mtmvn.mean, mean))
        self.assertTrue(approx_equal(mtmvn.variance, variance.view(2, 2)))
        self.assertTrue(torch.equal(mtmvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mtmvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
        mvn_times2 = mtmvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
        mvn_divby2 = mtmvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
        self.assertAlmostEqual(mtmvn.entropy().item(), 7.2648, places=4)
        self.assertAlmostEqual(mtmvn.log_prob(torch.zeros(2, 2, device=device)).item(), -7.3064, places=4)
        logprob = mtmvn.log_prob(torch.zeros(3, 2, 2, device=device))
        logprob_expected = -7.3064 * torch.ones(3, device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        conf_lower, conf_upper = mtmvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
        self.assertTrue(mtmvn.sample().shape == torch.Size([2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 2, 2]))

    def test_multitask_multivariate_normal_cuda(self):
        if torch.cuda.is_available():
            self.test_multitask_multivariate_normal(cuda=True)

    def test_multitask_multivariate_normal_batch(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float, device=device).repeat(2, 1, 1)
        variance = 1 + torch.arange(4, dtype=torch.float, device=device)
        covmat = torch.diag(variance).repeat(2, 1, 1)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)
        self.assertTrue(torch.equal(mtmvn.mean, mean))
        self.assertTrue(approx_equal(mtmvn.variance, variance.repeat(2, 1).view(2, 2, 2)))
        self.assertTrue(torch.equal(mtmvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mtmvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mtmvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mtmvn.covariance_matrix))
        mvn_times2 = mtmvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mtmvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mtmvn.covariance_matrix * 4))
        mvn_divby2 = mtmvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mtmvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mtmvn.covariance_matrix / 4))
        self.assertTrue(approx_equal(mtmvn.entropy(), 7.2648 * torch.ones(2, device=device)))
        logprob = mtmvn.log_prob(torch.zeros(2, 2, 2, device=device))
        logprob_expected = -7.3064 * torch.ones(2, device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        logprob = mtmvn.log_prob(torch.zeros(3, 2, 2, 2, device=device))
        logprob_expected = -7.3064 * torch.ones(3, 2, device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        conf_lower, conf_upper = mtmvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mtmvn.mean - 2 * mtmvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mtmvn.mean + 2 * mtmvn.stddev))
        self.assertTrue(mtmvn.sample().shape == torch.Size([2, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3])).shape == torch.Size([3, 2, 2, 2]))
        self.assertTrue(mtmvn.sample(torch.Size([3, 4])).shape == torch.Size([3, 4, 2, 2, 2]))

    def test_multitask_multivariate_normal_batch_cuda(self):
        if torch.cuda.is_available():
            self.test_multitask_multivariate_normal_batch(cuda=True)

    def test_multivariate_normal_correlated_sampels(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float, device=device)
        variance = 1 + torch.arange(4, dtype=torch.float, device=device)
        covmat = torch.diag(variance)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

        base_samples = mtmvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 2]))

        base_samples = mtmvn.get_base_samples()
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([2, 2]))

    def test_multivariate_normal_correlated_sampels_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_correlated_sampels(cuda=True)

    def test_multivariate_normal_batch_correlated_sampels(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([[0, 1], [2, 3]], dtype=torch.float, device=device).repeat(2, 1, 1)
        variance = 1 + torch.arange(4, dtype=torch.float, device=device)
        covmat = torch.diag(variance).repeat(2, 1, 1)
        mtmvn = MultitaskMultivariateNormal(mean=mean, covariance_matrix=covmat)

        base_samples = mtmvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 2, 2]))

        base_samples = mtmvn.get_base_samples()
        self.assertTrue(mtmvn.sample(base_samples=base_samples).shape == torch.Size([2, 2, 2]))

    def test_multivariate_normal_batch_correlated_sampels_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch_correlated_sampels(cuda=True)

    def test_log_prob(self):
        mean = torch.randn(4, 3)
        var = torch.randn(12).abs_()
        values = mean + 0.5
        diffs = (values - mean).view(-1)

        res = MultitaskMultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
        actual = -0.5 * (math.log(math.pi * 2) * 12 + var.log().sum() + (diffs / var * diffs).sum())
        self.assertLess((res - actual).div(res).abs().item(), 1e-2)

        mean = torch.randn(3, 4, 3)
        var = torch.randn(3, 12).abs_()
        values = mean + 0.5
        diffs = (values - mean).view(3, -1)

        res = MultitaskMultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
        actual = -0.5 * (math.log(math.pi * 2) * 12 + var.log().sum(-1) + (diffs / var * diffs).sum(-1))
        self.assertLess((res - actual).div(res).abs().norm(), 1e-2)


if __name__ == "__main__":
    unittest.main()
