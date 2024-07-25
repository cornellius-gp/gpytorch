#!/usr/bin/env python3

import unittest

from torch import Tensor

from gpytorch.priors import GammaPrior, HalfCauchyPrior, LogNormalPrior, NormalPrior


class TestPrior(unittest.TestCase):
    def test_state_dict(self):
        normal = NormalPrior(0.1, 1).state_dict()
        self.assertTrue("loc" in normal)
        self.assertTrue("scale" in normal)
        self.assertEqual(normal["loc"], 0.1)

        gamma = GammaPrior(1.1, 2).state_dict()
        self.assertTrue("concentration" in gamma)
        self.assertTrue("rate" in gamma)
        self.assertEqual(gamma["concentration"], 1.1)

        ln = LogNormalPrior(2.1, 1.2).state_dict()
        self.assertTrue("_transformed_loc" in ln)
        self.assertTrue("_transformed_scale" in ln)
        self.assertEqual(ln["_transformed_loc"], 2.1)

        hc = HalfCauchyPrior(1.3).state_dict()
        self.assertTrue("_transformed_scale" in hc)

    def test_load_state_dict(self):
        ln1 = LogNormalPrior(loc=0.5, scale=0.1)
        ln2 = LogNormalPrior(loc=2.5, scale=2.1)
        gm1 = GammaPrior(concentration=0.5, rate=0.1)
        gm2 = GammaPrior(concentration=2.5, rate=2.1)
        hc1 = HalfCauchyPrior(scale=1.1)
        hc2 = HalfCauchyPrior(scale=101.1)

        ln2.load_state_dict(ln1.state_dict())
        self.assertEqual(ln2.loc, ln1.loc)
        self.assertEqual(ln2.scale, ln1.scale)

        gm2.load_state_dict(gm1.state_dict())
        self.assertEqual(gm2.concentration, gm1.concentration)
        self.assertEqual(gm2.rate, gm1.rate)

        hc2.load_state_dict(hc1.state_dict())
        self.assertEqual(hc2.scale, hc1.scale)

    def test_transformed_attributes(self):
        norm = NormalPrior(loc=2.5, scale=2.1)
        ln = LogNormalPrior(loc=2.5, scale=2.1)
        hc = HalfCauchyPrior(scale=2.2)

        with self.assertRaisesRegex(AttributeError, "'NormalPrior' object has no attribute '_transformed_loc'"):
            getattr(norm, "_transformed_loc")

        self.assertTrue(getattr(ln, "_transformed_loc"), 2.5)
        norm.loc = Tensor([1.01])
        ln.loc = Tensor([1.01])
        self.assertEqual(ln._transformed_loc, 1.01)
        self.assertEqual(hc._transformed_scale, 2.2)
