#!/usr/bin/env python3

import unittest

import os
import random
import math
import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions import MultivariateNormal as TMultivariateNormal
from gpytorch.lazy import LazyTensor, NonLazyTensor, DiagLazyTensor
from test._utils import approx_equal


class TestMultivariateNormal(unittest.TestCase):
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

    def test_multivariate_normal_non_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=covmat, validate_args=True)
        self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
        self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
        self.assertTrue(approx_equal(mvn.variance, torch.diag(covmat)))
        self.assertTrue(approx_equal(mvn.scale_tril, covmat.sqrt()))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
        self.assertAlmostEqual(mvn.log_prob(torch.zeros(3, device=device)).item(), -4.8157, places=4)
        logprob = mvn.log_prob(torch.zeros(2, 3, device=device))
        logprob_expected = torch.tensor([-4.8157, -4.8157], device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_non_lazy_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_non_lazy(cuda=True)

    def test_multivariate_normal_batch_non_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=covmat.repeat(2, 1, 1), validate_args=True)
        self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
        self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
        self.assertTrue(approx_equal(mvn.variance, covmat.diag().repeat(2, 1)))
        self.assertTrue(approx_equal(mvn.scale_tril, torch.diag(covmat.diag().sqrt()).repeat(2, 1, 1)))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        self.assertTrue(approx_equal(mvn.entropy(), 4.3157 * torch.ones(2, device=device)))
        logprob = mvn.log_prob(torch.zeros(2, 3, device=device))
        logprob_expected = -4.8157 * torch.ones(2, device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        logprob = mvn.log_prob(torch.zeros(2, 2, 3, device=device))
        logprob_expected = -4.8157 * torch.ones(2, 2, device=device)
        self.assertTrue(approx_equal(logprob, logprob_expected))
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_batch_non_lazy_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch_non_lazy(cuda=True)

    def test_multivariate_normal_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))
        self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
        self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
        self.assertTrue(torch.equal(mvn.variance, torch.diag(covmat)))
        self.assertTrue(torch.equal(mvn.covariance_matrix, covmat))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
        # uses using root_decomposition which is not very reliable
        # self.assertAlmostEqual(mvn.entropy().item(), 4.3157, places=4)
        # self.assertAlmostEqual(mvn.log_prob(torch.zeros(3)).item(), -4.8157, places=4)
        # self.assertTrue(
        #     approx_equal(
        #         mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
        #     )
        # )
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 3]))

    def test_multivariate_normal_lazy_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_lazy(cuda=True)

    def test_multivariate_normal_batch_lazy(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=NonLazyTensor(covmat).repeat(2, 1, 1))
        self.assertTrue(torch.is_tensor(mvn.covariance_matrix))
        self.assertIsInstance(mvn.lazy_covariance_matrix, LazyTensor)
        self.assertTrue(approx_equal(mvn.variance, covmat.diag().repeat(2, 1)))
        mvn_plus1 = mvn + 1
        self.assertTrue(torch.equal(mvn_plus1.mean, mvn.mean + 1))
        self.assertTrue(torch.equal(mvn_plus1.covariance_matrix, mvn.covariance_matrix))
        mvn_times2 = mvn * 2
        self.assertTrue(torch.equal(mvn_times2.mean, mvn.mean * 2))
        self.assertTrue(torch.equal(mvn_times2.covariance_matrix, mvn.covariance_matrix * 4))
        mvn_divby2 = mvn / 2
        self.assertTrue(torch.equal(mvn_divby2.mean, mvn.mean / 2))
        self.assertTrue(torch.equal(mvn_divby2.covariance_matrix, mvn.covariance_matrix / 4))
        # TODO: Add tests for entropy, log_prob, etc. - this an issue b/c it
        # uses using root_decomposition which is not very reliable
        # self.assertTrue(approx_equal(mvn.entropy(), 4.3157 * torch.ones(2)))
        # self.assertTrue(
        #     approx_equal(mvn.log_prob(torch.zeros(2, 3)), -4.8157 * torch.ones(2))
        # )
        # self.assertTrue(
        #     approx_equal(mvn.log_prob(torch.zeros(2, 2, 3)), -4.8157 * torch.ones(2, 2))
        # )
        conf_lower, conf_upper = mvn.confidence_region()
        self.assertTrue(approx_equal(conf_lower, mvn.mean - 2 * mvn.stddev))
        self.assertTrue(approx_equal(conf_upper, mvn.mean + 2 * mvn.stddev))
        self.assertTrue(mvn.sample().shape == torch.Size([2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2])).shape == torch.Size([2, 2, 3]))
        self.assertTrue(mvn.sample(torch.Size([2, 4])).shape == torch.Size([2, 4, 2, 3]))

    def test_multivariate_normal_batch_lazy_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch_lazy(cuda=True)

    def test_multivariate_normal_correlated_sampels(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean, covariance_matrix=NonLazyTensor(covmat))

        base_samples = mvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 3]))

        base_samples = mvn.get_base_samples()
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3]))

    def test_multivariate_normal_correlated_sampels_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_correlated_sampels(cuda=True)

    def test_multivariate_normal_batch_correlated_sampels(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor([0, 1, 2], dtype=torch.float, device=device)
        covmat = torch.diag(torch.tensor([1, 0.75, 1.5], device=device))
        mvn = MultivariateNormal(mean=mean.repeat(2, 1), covariance_matrix=NonLazyTensor(covmat).repeat(2, 1, 1))

        base_samples = mvn.get_base_samples(torch.Size((3, 4)))
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([3, 4, 2, 3]))

        base_samples = mvn.get_base_samples()
        self.assertTrue(mvn.sample(base_samples=base_samples).shape == torch.Size([2, 3]))

    def test_multivariate_normal_batch_correlated_sampels_cuda(self):
        if torch.cuda.is_available():
            self.test_multivariate_normal_batch_correlated_sampels(cuda=True)

    def test_log_prob(self):
        mean = torch.randn(4)
        var = torch.randn(4).abs_()
        values = torch.randn(4)

        res = MultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
        actual = TMultivariateNormal(mean, torch.eye(4) * var).log_prob(values)
        self.assertLess((res - actual).div(res).abs().item(), 1e-2)

        mean = torch.randn(3, 4)
        var = torch.randn(3, 4).abs_()
        values = torch.randn(3, 4)

        res = MultivariateNormal(mean, DiagLazyTensor(var)).log_prob(values)
        actual = TMultivariateNormal(mean, var.unsqueeze(-1) * torch.eye(4).repeat(3, 1, 1)).log_prob(values)
        self.assertLess((res - actual).div(res).abs().norm(), 1e-2)

    def test_kl_divergence(self):
        mean0 = torch.randn(4)
        mean1 = mean0 + 1
        var0 = torch.randn(4).abs_()
        var1 = var0 * math.exp(2)

        dist_a = MultivariateNormal(mean0, DiagLazyTensor(var0))
        dist_b = MultivariateNormal(mean1, DiagLazyTensor(var0))
        dist_c = MultivariateNormal(mean0, DiagLazyTensor(var1))

        res = torch.distributions.kl.kl_divergence(dist_a, dist_a)
        actual = 0.0
        self.assertLess((res - actual).abs().item(), 1e-2)

        res = torch.distributions.kl.kl_divergence(dist_b, dist_a)
        actual = var0.reciprocal().sum().div(2.0)
        self.assertLess((res - actual).div(res).abs().item(), 1e-2)

        res = torch.distributions.kl.kl_divergence(dist_a, dist_c)
        actual = 0.5 * (8 - 4 + 4 * math.exp(-2))
        self.assertLess((res - actual).div(res).abs().item(), 1e-2)


if __name__ == "__main__":
    unittest.main()
