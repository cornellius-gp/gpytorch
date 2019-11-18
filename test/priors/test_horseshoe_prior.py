#!/usr/bin/env python3

import unittest

import torch

from gpytorch.priors import HorseshoePrior


class TestHorseshoePrior(unittest.TestCase):
    def test_horseshoe_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = HorseshoePrior(0.1).cuda()
            self.assertEqual(prior.scale.device.type, "cuda")

    def test_horseshoe_prior_validate_args(self):
        with self.assertRaises(ValueError):
            HorseshoePrior(-0.1, validate_args=True)

    def test_horseshoe_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        scale = torch.tensor(0.1, device=device)
        prior = HorseshoePrior(scale)
        t = torch.rand(3, device=device)
        prior.log_prob(t)

    def test_horseshoe_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            return self.test_horseshoe_prior_log_prob(cuda=True)

    def test_horseshoe_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        scale = torch.tensor(0.1, device=device)
        prior = HorseshoePrior(scale)
        prior_tf = HorseshoePrior(scale, transform=torch.exp)
        t = torch.tensor(0.5, device=device)
        self.assertTrue(torch.equal(prior_tf.log_prob(t), prior.log_prob(t.exp())))

    def test_horseshoe_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            return self.test_horseshoe_prior_log_prob_log_transform(cuda=True)

    def test_horseshoe_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")

        scale = torch.tensor([0.1, 0.25], device=device)
        prior = HorseshoePrior(scale)
        t = torch.ones(2, device=device)
        prior.log_prob(t)
        t = torch.ones(2, 2, device=device)
        prior.log_prob(t)
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(3, device=device))

        scale = torch.tensor([[0.1, 0.25], [0.3, 1.0]], device=device)
        prior = HorseshoePrior(scale)
        t = torch.ones(2, device=device)
        prior.log_prob(t)
        t = torch.ones(2, 2, device=device)
        prior.log_prob(t)
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(3, device=device))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(2, 3, device=device))

    def test_horseshoe_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            return self.test_horseshoe_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
