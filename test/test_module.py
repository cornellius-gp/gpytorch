#!/usr/bin/env python3

import unittest

import torch

from gpytorch.module import Module
from gpytorch.priors import NormalPrior
from gpytorch.test.base_test_case import BaseTestCase


class TestModule(BaseTestCase, unittest.TestCase):
    def test_named_priors_with_shared_modules(self):
        prior = NormalPrior(0.0, 1.0)
        shared = Module()
        shared.register_parameter("value", torch.nn.Parameter(torch.tensor(0.3)))
        shared.register_prior("value_prior", prior, "value")
        root = Module()
        root.shared = shared
        root.branch = Module()
        root.branch.shared = shared
        root.other = Module()
        root.other.register_parameter("value", torch.nn.Parameter(torch.tensor(0.7)))
        # The same prior object may legitimately govern a different parameter.
        root.other.register_prior("value_prior", prior, "value")
        priors = list(root.named_priors())
        self.assertEqual([p[0] for p in priors], ["shared.value_prior", "other.value_prior"])
        log_prob = sum(p.log_prob(closure(parent)) for _, parent, p, closure, _ in priors)
        gradients = torch.autograd.grad(log_prob, (shared.value, root.other.value))
        self.assertTrue(torch.allclose(gradients[0], -shared.value))
        self.assertTrue(torch.allclose(gradients[1], -root.other.value))

    def test_register_prior(self):
        mock_prior = Module()
        m = Module()
        with self.assertRaises(AttributeError):
            m.register_prior("mock_prior", mock_prior, "foo")
        m.register_parameter("foo", torch.nn.Parameter(torch.zeros(1)))
        with self.assertRaises(RuntimeError):
            m.register_prior("mock_prior", mock_prior, "foo", unittest.mock.Mock())
        m.register_prior("mock_prior", mock_prior, "foo")
        with self.assertRaises(ValueError):
            m.register_prior("mock_prior", mock_prior, lambda: None)
        m.register_prior("mock_prior", mock_prior, lambda m: None)
        with self.assertRaises(ValueError):
            m.register_prior("mock_prior", mock_prior, lambda m: None, lambda v: None)
        m.register_prior("mock_prior", mock_prior, lambda m: None, lambda m, v: None)
