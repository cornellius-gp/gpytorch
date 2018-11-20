#!/usr/bin/env python3

import unittest

import torch
from gpytorch.priors import SmoothedBoxPrior
from test._utils import approx_equal


class TestSmoothedBoxPrior(unittest.TestCase):
    def test_smoothed_box_prior_to_gpu(self):
        if torch.cuda.is_available():
            prior = SmoothedBoxPrior(torch.zeros(2), torch.ones(2)).cuda()
            self.assertEqual(prior.a.device.type, "cuda")
            self.assertEqual(prior.b.device.type, "cuda")
            self.assertEqual(prior.sigma.device.type, "cuda")
            self.assertEqual(prior._c.device.type, "cuda")
            self.assertEqual(prior._r.device.type, "cuda")
            self.assertEqual(prior._M.device.type, "cuda")
            self.assertEqual(prior.tails.loc.device.type, "cuda")
            self.assertEqual(prior.tails.scale.device.type, "cuda")

    def test_smoothed_box_prior_validate_args(self):
        with self.assertRaises(ValueError):
            SmoothedBoxPrior(torch.ones(2), torch.zeros(2), validate_args=True)

    def test_smoothed_box_prior_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        a, b = torch.zeros(2, device=device), torch.ones(2, device=device)
        sigma = 0.1
        prior = SmoothedBoxPrior(a, b, sigma)

        self.assertTrue(torch.equal(prior.a, a))
        self.assertTrue(torch.equal(prior.b, b))
        self.assertTrue(torch.equal(prior.sigma, torch.full_like(prior.a, sigma)))
        self.assertTrue(torch.all(approx_equal(prior._M, torch.full_like(prior.a, 1.6073))))

        t = torch.tensor([0.5, 1.1], device=device)
        self.assertAlmostEqual(prior.log_prob(t).item(), -0.9473, places=4)
        t = torch.tensor([[0.5, 1.1], [0.1, 0.25]], device=device)
        log_prob_expected = torch.tensor([-0.947347, -0.447347], device=t.device)
        self.assertTrue(torch.all(approx_equal(prior.log_prob(t), log_prob_expected)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.zeros(3, device=device))

    def test_smoothed_box_prior_log_prob_cuda(self):
        if torch.cuda.is_available():
            return self.test_smoothed_box_prior_log_prob(cuda=True)

    def test_smoothed_box_prior_log_prob_log_transform(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        a, b = torch.zeros(2, device=device), torch.ones(2, device=device)
        sigma = 0.1
        prior = SmoothedBoxPrior(a, b, sigma, transform=torch.exp)

        t = torch.tensor([0.5, 1.1], device=device).log()
        self.assertAlmostEqual(prior.log_prob(t).item(), -0.9473, places=4)
        t = torch.tensor([[0.5, 1.1], [0.1, 0.25]], device=device).log()
        log_prob_expected = torch.tensor([-0.947347, -0.447347], device=t.device)
        self.assertTrue(torch.all(approx_equal(prior.log_prob(t), log_prob_expected)))
        with self.assertRaises(RuntimeError):
            prior.log_prob(torch.ones(3, device=device))

    def test_smoothed_box_prior_log_prob_log_transform_cuda(self):
        if torch.cuda.is_available():
            return self.test_smoothed_box_prior_log_prob_log_transform(cuda=True)

    def test_smoothed_box_prior_batch_log_prob(self, cuda=False):
        # TODO: Implement test for batch mode
        pass

    def test_smoothed_box_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            return self.test_smoothed_box_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
