#!/usr/bin/env python3

import unittest

import torch

from gpytorch.module import Module
from gpytorch.test.base_test_case import BaseTestCase


class TestModule(BaseTestCase, unittest.TestCase):
    def test_apply_cache_and_recurse(self):
        class CachedModule(Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("value", torch.tensor(1.0))
                self.cache = self.value.clone()

            def _clear_cache(self):
                self.cache = None

        module = CachedModule()
        module.child = CachedModule()
        self.assertIs(module._apply(lambda tensor: tensor, recurse=False), module)
        self.assertIsNone(module.cache)
        self.assertIsNotNone(module.child.cache)
        module.double()
        self.assertIsNone(module.child.cache)
        self.assertEqual(module.value.dtype, torch.float64)
        self.assertEqual(module.child.value.dtype, torch.float64)
        self.assertIs(Module().to(dtype=torch.float64).__class__, Module)

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
