#!/usr/bin/env python3

import unittest

import torch

from gpytorch.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfCauchyPrior,
    LogNormalPrior,
    NormalPrior,
)
from gpytorch.priors.utils import _load_transformed_to_base_dist


class TestUtils(unittest.TestCase):
    def test_bufferize_attributes(self):
        prior = NormalPrior(0.01, 1)
        self.assertEqual(prior.loc, 0.01)
        self.assertEqual(prior.loc.requires_grad, False)

        prior = GammaPrior(1, 2)
        self.assertEqual(prior.rate, 2.0)
        self.assertEqual(prior.rate.requires_grad, False)

        prior = LogNormalPrior(2.1, 1.2)
        self.assertEqual(prior._transformed_loc, 2.1)
        self.assertEqual(prior.loc, 2.1)
        self.assertEqual(prior._transformed_scale, 1.2)
        self.assertEqual(prior._transformed_scale.requires_grad, False)

        prior = HalfCauchyPrior(1.3)
        self.assertEqual(prior._transformed_scale, 1.3)
        self.assertEqual(prior._transformed_scale.requires_grad, False)

    def test_load_transformed_to_base_dist(self):
        lognormal = LogNormalPrior(loc=2.5, scale=2.1)
        self.assertEqual(lognormal._transformed_loc, 2.5)
        self.assertEqual(lognormal._transformed_scale, 2.1)

        lognormal._transformed_loc = torch.Tensor([0.11])
        lognormal._transformed_scale = torch.Tensor([11])
        _load_transformed_to_base_dist(lognormal)
        self.assertEqual(lognormal._transformed_loc, 0.11)
        self.assertEqual(lognormal._transformed_scale, 11)
        self.assertEqual(lognormal.loc, 0.11)
        self.assertEqual(lognormal.scale, 11)

        halfcauchy = HalfCauchyPrior(scale=11.1)
        self.assertEqual(halfcauchy._transformed_scale, 11.1)
        halfcauchy._transformed_scale = torch.Tensor([0.11])
        _load_transformed_to_base_dist(halfcauchy)
        self.assertEqual(halfcauchy._transformed_scale, 0.11)
        self.assertEqual(halfcauchy.scale, 0.11)
