#!/usr/bin/env python3

import unittest

import torch
from torch import Tensor

from gpytorch.priors import (
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
    UniformPrior,
)
from gpytorch.priors.utils import BUFFERED_PREFIX


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
        self.assertTrue(f"{BUFFERED_PREFIX}loc" in ln)
        self.assertTrue(f"{BUFFERED_PREFIX}scale" in ln)
        self.assertEqual(ln[f"{BUFFERED_PREFIX}loc"], 2.1)

        hc = HalfCauchyPrior(1.3).state_dict()
        self.assertTrue(f"{BUFFERED_PREFIX}scale" in hc)

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

        with self.assertRaisesRegex(
            AttributeError,
            f"'NormalPrior' object has no attribute '{BUFFERED_PREFIX}loc'",
        ):
            getattr(norm, f"{BUFFERED_PREFIX}loc")

        self.assertEqual(getattr(ln, f"{BUFFERED_PREFIX}loc"), 2.5)
        norm.loc = Tensor([1.01])
        ln.loc = Tensor([1.01])
        self.assertEqual(getattr(ln, f"{BUFFERED_PREFIX}loc"), 1.01)
        self.assertEqual(getattr(hc, f"{BUFFERED_PREFIX}scale"), 2.2)

    def test_transformed_priors_move_with_to(self):
        # TransformedDistribution priors must move base_dist params with .to(),
        # not only the registered _buffered_* copies (see #2581).
        device = torch.device("cpu")
        priors = [
            HalfCauchyPrior(1.0),
            HalfNormalPrior(1.0),
            LogNormalPrior(1.0, 1.0),
            UniformPrior(1.0, 2.0),
            NormalPrior(1.0, 1.0),
            GammaPrior(1.0, 1.0),
        ]
        for prior in priors:
            prior = prior.to(device)
            samples = prior.rsample()
            self.assertEqual(samples.device.type, device.type)
            for value in prior.state_dict().values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device.type, device.type)
            base = getattr(prior, "base_dist", None)
            if base is not None:
                for name in ("loc", "scale", "low", "high", "concentration", "rate"):
                    if hasattr(base, name):
                        tensor = getattr(base, name)
                        if torch.is_tensor(tensor):
                            self.assertEqual(
                                tensor.device.type,
                                device.type,
                                msg=f"{type(prior).__name__}.base_dist.{name}",
                            )

    def test_transformed_priors_move_with_to_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda:0")
        for prior in (
            HalfCauchyPrior(1.0),
            HalfNormalPrior(1.0),
            LogNormalPrior(0.5, 0.5),
            UniformPrior(1.0, 2.0),
        ):
            prior = prior.to(device)
            samples = prior.rsample()
            self.assertEqual(samples.device.type, "cuda")
            for value in prior.state_dict().values():
                if torch.is_tensor(value):
                    self.assertEqual(value.device.type, "cuda")
            base = getattr(prior, "base_dist", None)
            if base is not None:
                for name in ("loc", "scale"):
                    if hasattr(base, name):
                        tensor = getattr(base, name)
                        if torch.is_tensor(tensor):
                            self.assertEqual(tensor.device.type, "cuda")

    def test_uniform_prior_state_dict_buffers(self):
        prior = UniformPrior(1.0, 2.0)
        state = prior.state_dict()
        self.assertIn("low", state)
        self.assertIn("high", state)
        self.assertEqual(state["low"], 1.0)
        self.assertEqual(state["high"], 2.0)
