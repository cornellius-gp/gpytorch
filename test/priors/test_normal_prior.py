from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import unittest
import torch
from gpytorch.priors import NormalPrior


class TestNormalPrior(unittest.TestCase):
    def test_scalar_normal_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            NormalPrior(0, -1)

    def test_scalar_normal_prior(self):
        prior = NormalPrior(0, 1)
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(torch.rand(1)))
        self.assertEqual(prior.shape, torch.Size([1]))
        self.assertEqual(prior.loc.item(), 0.0)
        self.assertEqual(prior.scale.item(), 1.0)
        self.assertAlmostEqual(
            prior.log_prob(prior.loc.new([0.0])).item(), math.log(1 / math.sqrt(2 * math.pi)), places=5
        )

    def test_scalar_normal_prior_log_transform(self):
        prior = NormalPrior(0, 1, log_transform=True)
        self.assertTrue(prior.log_transform)
        self.assertAlmostEqual(
            prior.log_prob(prior.loc.new([0.0])).item(), math.log(1 / math.sqrt(2 * math.pi) * math.exp(-0.5)), places=5
        )

    def test_vector_normal_prior_invalid_params(self):
        with self.assertRaises(ValueError):
            NormalPrior(torch.tensor([-0.5, 0.5]), torch.tensor([-0.1, 1.0]))

    def test_vector_normal_prior_size(self):
        prior = NormalPrior(0, 1, size=2)
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(prior.loc.new_zeros(1)))
        self.assertEqual(prior.shape, torch.Size([2]))
        self.assertTrue(torch.equal(prior.loc, prior.loc.new([0.0, 0.0])))
        self.assertTrue(torch.equal(prior.scale, prior.scale.new([1.0, 1.0])))
        parameter = prior.loc.new([1.0, 2.0])
        self.assertAlmostEqual(
            prior.log_prob(parameter).item(),
            2 * math.log(1 / math.sqrt(2 * math.pi)) - 0.5 * (parameter ** 2).sum().item(),
            places=5,
        )

    def test_vector_normal_prior(self):
        prior = NormalPrior(torch.tensor([-0.5, 0.5]), torch.tensor([0.5, 1.0]))
        self.assertFalse(prior.log_transform)
        self.assertTrue(prior.is_in_support(torch.rand(1)))
        self.assertEqual(prior.shape, torch.Size([2]))
        self.assertTrue(torch.equal(prior.loc, prior.loc.new([-0.5, 0.5])))
        self.assertTrue(torch.equal(prior.scale, prior.scale.new([0.5, 1.0])))
        parameter = prior.loc.new([1.0, 2.0])
        expected_log_prob = (
            (
                (1 / math.sqrt(2 * math.pi) / prior.scale).log()
                - 0.5 / prior.scale ** 2 * (prior.loc.new_tensor(parameter) - prior.loc) ** 2
            )
            .sum()
            .item()
        )
        self.assertAlmostEqual(prior.log_prob(prior.loc.new_tensor(parameter)).item(), expected_log_prob, places=5)


if __name__ == "__main__":
    unittest.main()
