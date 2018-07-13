from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import torch
from gpytorch.priors import GammaPrior


class TestGammaPrior(unittest.TestCase):
    def test_scalar_gamma_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            GammaPrior(0, 1)
        with self.assertRaises(ValueError):
            GammaPrior(1, 0)

    def test_scalar_gamma_prior(self):
        prior = GammaPrior(1, 1)  # this is an exponential w/ rate 1
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(prior.rate.new([1])))
        self.assertFalse(prior.is_in_support(prior.rate.new([-1])))
        self.assertEqual(prior.shape, torch.Size([1]))
        self.assertEqual(prior.concentration.item(), 1.0)
        self.assertEqual(prior.rate.item(), 1.0)
        self.assertAlmostEqual(prior.log_prob(prior.rate.new([1.0])).item(), -1.0, places=5)

    def test_scalar_gamma_prior_log_transform(self):
        prior = GammaPrior(1, 1, log_transform=True)
        self.assertTrue(prior.log_transform)
        self.assertAlmostEqual(prior.log_prob(prior.rate.new([0.0])).item(), -1.0, places=5)

    def test_vector_gamma_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            GammaPrior(torch.tensor([-0.5, 0.5]), torch.tensor([1.0, 1.0]))
        with self.assertRaises(ValueError):
            GammaPrior(torch.tensor([0.5, 0.5]), torch.tensor([-0.1, 1.0]))

    def test_vector_gamma_prior_size(self):
        prior = GammaPrior(1, 1, size=2)
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(prior.rate.new_ones(2)))
        self.assertFalse(prior.is_in_support(prior.rate.new_zeros(2)))
        self.assertEqual(prior.shape, torch.Size([2]))
        self.assertTrue(torch.equal(prior.concentration, prior.rate.new([1.0, 1.0])))
        self.assertTrue(torch.equal(prior.rate, prior.rate.new([1.0, 1.0])))
        parameter = prior.rate.new([1.0, 2.0])
        self.assertAlmostEqual(prior.log_prob(parameter).item(), -3.0, places=5)

    def test_vector_gamma_prior(self):
        prior = GammaPrior(torch.tensor([1.0, 2.0]), torch.tensor([0.5, 2.0]))
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(torch.rand(1)))
        self.assertEqual(prior.shape, torch.Size([2]))
        self.assertTrue(torch.equal(prior.concentration, prior.rate.new([1.0, 2.0])))
        self.assertTrue(torch.equal(prior.rate, prior.rate.new([0.5, 2.0])))
        parameter = prior.rate.new([1.0, math.exp(1)])
        expected_log_prob = prior.rate.new([math.log(0.5) - 0.5, 2 * math.log(2) + 1 - 2 * math.exp(1)]).sum().item()
        self.assertAlmostEqual(prior.log_prob(prior.rate.new_tensor(parameter)).item(), expected_log_prob, places=5)


if __name__ == "__main__":
    unittest.main()
