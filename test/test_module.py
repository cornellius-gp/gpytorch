#!/usr/bin/env python3

import unittest

import torch

from gpytorch.module import Module
from gpytorch.test.base_test_case import BaseTestCase


class TestModule(BaseTestCase, unittest.TestCase):
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
