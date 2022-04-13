#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import HalfCauchy

from gpytorch.priors import HalfCauchyPrior
from gpytorch.test.utils import least_used_cuda_device


class TestHalfCauchyPrior(unittest.TestCase):
    def test_half_cauchy_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = HalfCauchy(1.0).cuda()
            self.assertEqual(prior.concentration.device.type, "cuda")
            self.assertEqual(prior.rate.device.type, "cuda")

    def test_half_cauchy_prior_validate_args(self):
        with self.assertRaises(ValueError):
            HalfCauchyPrior(-1, validate_args=True)
        with self.assertRaises(ValueError):
            HalfCauchyPrior(-1, validate_args=True)

    def test_half_cauchy_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfCauchyPrior(0.1)
        dist = HalfCauchy(0.1)

        t = torch.tensor(1.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([1.5, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([[1.0, 0.5], [3.0, 0.25]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

    def test_half_cauchy_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_gamma_prior_log_prob(cuda=True)

    def test_half_cauchy_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfCauchyPrior(0.1, transform=torch.exp)
        dist = HalfCauchy(0.1)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))

    def test_half_cauchy_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_half_cauchy_prior_log_prob_log_transform(cuda=True)

    def test_half_cauchy_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfCauchyPrior(0.1)
        dist = HalfCauchy(0.1)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

        scale = torch.tensor([0.1, 1.0], device=device)
        prior = HalfCauchyPrior(scale)
        dist = HalfCauchy(scale)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.ones(3, device=device))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.ones(2, 3, device=device))

    def test_half_cauchy_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_half_cauchy_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
