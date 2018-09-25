from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from gpytorch.priors import LKJPrior
from gpytorch.utils import approx_equal


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

        self.assertFalse(prior.log_transform)
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
            return self.test_lkj_prior_log_prob(cuda=True)

    def test_lkj_prior_batch_log_prob(self, cuda=False):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        prior = LKJPrior(2, torch.tensor([0.5, 1.5], device=device))

        self.assertFalse(prior.log_transform)
        S = torch.eye(2, device=device)
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-1.86942, -0.483129], device=S.device)))
        S = torch.stack([S, torch.tensor([[1.0, 0.5], [0.5, 1]], device=S.device)])
        self.assertTrue(approx_equal(prior.log_prob(S), torch.tensor([-1.86942, -0.62697], device=S.device)))
        with self.assertRaises(ValueError):
            prior.log_prob(torch.eye(3, device=device))

    def test_lkj_prior_batch_log_prob_cuda(self):
        if torch.cuda.is_available():
            return self.test_lkj_prior_batch_log_prob(cuda=True)


if __name__ == "__main__":
    unittest.main()
