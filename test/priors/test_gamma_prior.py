#!/usr/bin/env python3

import unittest

import torch
from torch.distributions import Gamma

from gpytorch.priors import GammaPrior
from gpytorch.test.utils import least_used_cuda_device


class TestGammaPrior(unittest.TestCase):
    def test_gamma_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = GammaPrior(1.0, 1.0).cuda()
            self.assertEqual(prior.concentration.device.type, "cuda")
            self.assertEqual(prior.rate.device.type, "cuda")

    def test_gamma_prior_validate_args(self):
        with self.assertRaises(ValueError):
            GammaPrior(0, 1, validate_args=True)
        with self.assertRaises(ValueError):
            GammaPrior(1, 0, validate_args=True)

    def test_gamma_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        concentration = torch.tensor(1.0, device=device)
        rate = torch.tensor(1.0, device=device)
        prior = GammaPrior(concentration, rate)
        dist = Gamma(concentration, rate)

        t = torch.tensor(1.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([1.5, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.tensor([[1.0, 0.5], [3.0, 0.25]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))

    def test_gamma_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_gamma_prior_log_prob(cuda=True)

    def test_gamma_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        concentration = torch.tensor(1.0, device=device)
        rate = torch.tensor(1.0, device=device)
        prior = GammaPrior(concentration, rate, transform=torch.exp)
        dist = Gamma(concentration, rate)

        t = torch.tensor(0.0, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([-1, 0.5], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))
        t = torch.tensor([[-1, 0.5], [0.1, -2.0]], device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t.exp())))

    def test_gamma_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_gamma_prior_log_prob_log_transform(cuda=True)

    def test_gamma_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        concentration = torch.tensor([1.0, 2.0], device=device)
        rate = torch.tensor([1.0, 2.0], device=device)
        prior = GammaPrior(concentration, rate)
        dist = Gamma(concentration, rate)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(3, device=device))

        mean = torch.tensor([[1.0, 2.0], [0.5, 3.0]], device=device)
        variance = torch.tensor([[1.0, 2.0], [0.5, 1.0]], device=device)
        prior = GammaPrior(mean, variance)
        dist = Gamma(mean, variance)
        t = torch.ones(2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        t = torch.ones(2, 2, device=device)
        self.assertTrue(torch.equal(prior.log_prob(t), dist.log_prob(t)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(3, device=device))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(2, 3, device=device))

    def test_gamma_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            with least_used_cuda_device():
                return self.test_gamma_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
