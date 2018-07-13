from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import torch
from gpytorch.priors import InverseWishartPrior, WishartPrior
from gpytorch.priors.wishart_prior import log_mv_gamma


class TestWishartPrior(unittest.TestCase):

    def setUp(self):
        self.nu = 3
        self.K = torch.Tensor([[1, 0], [0, 1]])
        self.K_bad = torch.Tensor([[1, 1], [1, 1]])

    def test_wishart_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            WishartPrior(1, self.K)
        with self.assertRaises(ValueError):
            WishartPrior(3, self.K_bad)

    def test_wishart_prior(self):
        prior = WishartPrior(self.nu, self.K)
        self.assertFalse(prior.log_transform)
        self.assertEqual(prior.shape, self.K.shape)
        self.assertTrue(torch.equal(prior.K_inv, torch.inverse(self.K)))
        self.assertEqual(prior.nu, self.nu)
        self.assertEqual(prior.C, - (3 * math.log(2) + log_mv_gamma(2, 3 / 2)))
        self.assertTrue(prior.is_in_support(self.K))
        self.assertFalse(prior.is_in_support(self.K_bad))
        self.assertAlmostEqual(prior.log_prob(self.K).item(), -3.531024, places=4)


class TestInverseWishartPrior(unittest.TestCase):

    def setUp(self):
        self.nu = 1
        self.K = torch.Tensor([[1, 0], [0, 1]])
        self.K_bad = torch.Tensor([[1, 1], [1, 1]])

    def test_inverse_wishart_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            InverseWishartPrior(0, self.K)
        with self.assertRaises(ValueError):
            InverseWishartPrior(1, self.K_bad)

    def test_inverse_wishart_prior(self):
        prior = InverseWishartPrior(self.nu, self.K)
        self.assertFalse(prior.log_transform)
        self.assertEqual(prior.shape, self.K.shape)
        self.assertTrue(torch.equal(prior.K, self.K))
        self.assertEqual(prior.nu, self.nu)
        self.assertEqual(prior.C, -2 * math.log(2) - log_mv_gamma(2, 1))
        self.assertTrue(prior.is_in_support(self.K))
        self.assertFalse(prior.is_in_support(self.K_bad))
        self.assertAlmostEqual(prior.log_prob(self.K).item(), -3.531024, places=4)


if __name__ == "__main__":
    unittest.main()
