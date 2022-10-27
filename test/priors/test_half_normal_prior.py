#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import HalfNormal

from gpytorch.priors import HalfNormalPrior
from gpytorch.test.utils import least_used_cuda_device


class TestHalfNormalPrior(unittest.TestCase):
    def test_half_normal_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = HalfNormalPrior(1.0).cuda()
            self.assertEqual(prior.concentration.device.type, "cuda")
            self.assertEqual(prior.rate.device.type, "cuda")

    def test_half_normal_prior_validate_args(self):
        with self.assertRaises(ValueError):
            HalfNormalPrior(-1, validate_args=True)

    def test_half_normal_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfNormalPrior(1.0)
        dist = HalfNormalPrior(1.0)

        t = torch.tensor(1.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([1.5, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([[1.0, 0.5], [3.0, 0.25]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

    def test_half_normal_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_half_normal_prior_log_prob(cuda=True)

    def test_half_normal_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfNormalPrior(1.0, transform=torch.exp)
        dist = HalfNormalPrior(1.0)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))

    def test_half_normal_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_half_normal_prior_log_prob_log_transform(cuda=True)

    def test_half_normal_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = HalfNormalPrior(1.0)
        dist = HalfNormal(1.0)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

        scale = torch.tensor([1.0, 10.0], device=device)
        prior = HalfNormalPrior(scale)
        dist = HalfNormal(scale)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.ones(3, device=device))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.ones(2, 3, device=device))

    def test_half_normal_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_half_normal_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
