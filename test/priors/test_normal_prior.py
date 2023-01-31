#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Normal

from gpytorch.priors import NormalPrior
from gpytorch.test.utils import least_used_cuda_device


class TestNormalPrior(unittest.TestCase):
    def test_normal_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = NormalPrior(0, 1).cuda()
            self.assertEqual(prior.loc.device.type, "cuda")
            self.assertEqual(prior.scale.device.type, "cuda")

    def test_normal_prior_validate_args(self):
        with self.assertRaises(ValueError):
            NormalPrior(0, -1, validate_args=True)

    def test_normal_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor(0.0, device=device)
        variance = torch.tensor(1.0, device=device)
        prior = NormalPrior(mean, variance)
        dist = Normal(mean, variance)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

    def test_normal_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_normal_prior_log_prob(cuda=True)

    def test_normal_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        mean = torch.tensor(0.0, device=device)
        variance = torch.tensor(1.0, device=device)
        prior = NormalPrior(mean, variance, transform=torch.exp)
        dist = Normal(mean, variance)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))

    def test_normal_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_normal_prior_log_prob_log_transform(cuda=True)

    def test_normal_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        mean = torch.tensor([0.0, 1.0], device=device)
        variance = torch.tensor([1.0, 2.0], device=device)
        prior = NormalPrior(mean, variance)
        dist = Normal(mean, variance)
        t = torch.zeros(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.zeros(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(3, device=device))

        mean = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], device=device)
        variance = torch.tensor([[1.0, 2.0], [0.5, 1.0]], device=device)
        prior = NormalPrior(mean, variance)
        dist = Normal(mean, variance)
        t = torch.zeros(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.zeros(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(3, device=device))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(2, 3, device=device))

    def test_normal_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_normal_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
