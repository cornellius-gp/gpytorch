#!/usr/bin/env python3

import unittest
from math import exp

import torch

from gpytorch.priors import LKJCholeskyFactorPrior, LKJCovariancePrior, LKJPrior, SmoothedBoxPrior
from gpytorch.test.utils import approx_equal, least_used_cuda_device


class TestLKJPrior(unittest.TestCase):
    def test_lkj_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = LKJPrior(2, 1.0).cuda()
            self.assertEqual(prior.eta.device.type, "cuda")
            self.assertEqual(prior.C.device.type, "cuda")

    def test_lkj_prior_validate_args(self):
        LKJPrior(2, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJPrior(1.5, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJPrior(2, -1.0, validate_args=True)

    def test_lkj_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJPrior(2, torch.tensor(0.5, device=device))

        S = torch.eye(2, device=device)
        self.assertAlmostEqual(prior.log_prob(S).item(), -1.86942, places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-1.86942, -1.72558], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

        # For eta=1.0 log_prob is flat over all covariance matrices
        prior = LKJPrior(2, torch.tensor(1.0, device=device))
        self.assertTrue(torch.all(prior.log_prob(S) == prior.C))

    def test_lkj_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_prior_log_prob(cuda=True)

    def test_lkj_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJPrior(2, torch.tensor([0.5, 1.5], device=device))

        S = torch.eye(2, device=device)
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-1.86942, -0.483129], device=S.device)))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-1.86942, -0.62697], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_prior_batch_log_prob(cuda=True)


class TestLKJCholeskyFactorPrior(unittest.TestCase):
    def test_lkj_cholesky_factor_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = LKJCholeskyFactorPrior(2, 1.0).cuda()
            self.assertEqual(prior.eta.device.type, "cuda")
            self.assertEqual(prior.C.device.type, "cuda")

    def test_lkj_cholesky_factor_prior_validate_args(self):
        LKJCholeskyFactorPrior(2, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCholeskyFactorPrior(1.5, 1.0, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCholeskyFactorPrior(2, -1.0, validate_args=True)

    def test_lkj_cholesky_factor_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJCholeskyFactorPrior(2, torch.tensor(0.5, device=device))
        S = torch.eye(2, device=device)
        S_chol = torch.cholesky(S)
        self.assertAlmostEqual(prior.log_prob(S_chol).item(), -1.86942, places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S_chol.device)])
        S_chol = torch.stack([torch.cholesky(Si) for Si in S])
        self.assertTrue(approx_equal(prior.log_prob(S_chol), torch.tensor([-1.86942, -1.72558], device=S_chol.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

        # For eta=1.0 log_prob is flat over all covariance matrices
        prior = LKJCholeskyFactorPrior(2, torch.tensor(1.0, device=device))
        self.assertTrue(torch.all(prior.log_prob(S_chol) == prior.C))

    def test_lkj_cholesky_factor_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_cholesky_factor_prior_log_prob(cuda=True)

    def test_lkj_cholesky_factor_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJCholeskyFactorPrior(2, torch.tensor([0.5, 1.5], device=device))

        S = torch.eye(2, device=device)
        S_chol = torch.cholesky(S)
        self.assertTrue(approx_equal(prior.log_prob(S_chol), torch.tensor([-1.86942, -0.483129], device=S_chol.device)))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        S_chol = torch.stack([torch.cholesky(Si) for Si in S])
        self.assertTrue(approx_equal(prior.log_prob(S_chol), torch.tensor([-1.86942, -0.62697], device=S_chol.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_cholesky_factor_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_cholesky_factor_prior_batch_log_prob(cuda=True)


class TestLKJCovariancePrior(unittest.TestCase):
    def test_lkj_covariance_prior_to_gpu(self):
        if torch.cuda.is_available():
            sd_prior = SmoothedBoxPrior(exp(-1), exp(1))
            prior = LKJCovariancePrior(2, 1.0, sd_prior).cuda()
            self.assertEqual(prior.correlation_prior.eta.device.type, "cuda")
            self.assertEqual(prior.correlation_prior.C.device.type, "cuda")
            self.assertEqual(prior.sd_prior.a.device.type, "cuda")

    def test_lkj_covariance_prior_validate_args(self):
        sd_prior = SmoothedBoxPrior(exp(-1), exp(1), validate_args=True)
        LKJCovariancePrior(2, 1.0, sd_prior)
        with self.assertRaises(ValueError):
            LKJCovariancePrior(1.5, 1.0, sd_prior, validate_args=True)
        with self.assertRaises(ValueError):
            LKJCovariancePrior(2, -1.0, sd_prior, validate_args=True)

    def test_lkj_covariance_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        sd_prior = SmoothedBoxPrior(exp(-1), exp(1))
        if cuda:
            sd_prior = sd_prior.cuda()
        prior = LKJCovariancePrior(2, torch.tensor(0.5, device=device), sd_prior)
        S = torch.eye(2, device=device)
        self.assertAlmostEqual(prior.log_prob(S).item(), -3.59981, places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-3.59981, -3.45597], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

        # For eta=1.0 log_prob is flat over all covariance matrices
        prior = LKJCovariancePrior(2, torch.tensor(1.0, device=device), sd_prior)
        marginal_sd = torch.diagonal(S, dim1=-2, dim2=-1).sqrt()
        log_prob_expected = prior.correlation_prior.C + prior.sd_prior.log_prob(marginal_sd)
        self.assertTrue(approx_equal(prior.log_prob(S), log_prob_expected))

    def test_lkj_covariance_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_covariance_prior_log_prob(cuda=True)

    def test_lkj_covariance_prior_log_prob_hetsd(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        a = torch.tensor([exp(-1), exp(-2)], device=device)
        b = torch.tensor([exp(1), exp(2)], device=device)
        sd_prior = SmoothedBoxPrior(a, b)
        prior = LKJCovariancePrior(2, torch.tensor(0.5, device=device), sd_prior)
        S = torch.eye(2, device=device)
        self.assertAlmostEqual(prior.log_prob(S).item(), -4.71958, places=4)
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-4.71958, -4.57574], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

        # For eta=1.0 log_prob is flat over all covariance matrices
        prior = LKJCovariancePrior(2, torch.tensor(1.0, device=device), sd_prior)
        marginal_sd = torch.diagonal(S, dim1=-2, dim2=-1).sqrt()
        log_prob_expected = prior.correlation_prior.C + prior.sd_prior.log_prob(marginal_sd)
        self.assertTrue(approx_equal(prior.log_prob(S), log_prob_expected))

    def test_lkj_covariance_prior_log_prob_hetsd_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_covariance_prior_log_prob_hetsd(cuda=True)

    def test_lkj_covariance_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        v = torch.ones(2, 1, device=device)
        sd_prior = SmoothedBoxPrior(exp(-1) * v, exp(1) * v)
        prior = LKJCovariancePrior(2, torch.tensor([0.5, 1.5], device=device), sd_prior)

        S = torch.eye(2, device=device)
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-3.59981, -2.21351], device=S.device)))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-3.59981, -2.35735], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_covariance_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                self.test_lkj_covariance_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
