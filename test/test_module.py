#!/usr/bin/env python3

import unittest

import torch

from gpytorch.module import Module
from gpytorch.test.base_test_case import BaseTestCase


class TestModule(BaseTestCase, unittest.TestCase):
    def test_initialize_float_with_transformed_constraint(self):
        from gpytorch.constraints import GreaterThan, Interval

        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for device in devices:
            for dtype in (torch.float, torch.double):
                for constraint_type in (GreaterThan, Interval):
                    with self.subTest(device=device, dtype=dtype, constraint=constraint_type):
                        module = Module()
                        parameter = torch.nn.Parameter(torch.zeros(2, dtype=dtype, device=device))
                        module.register_parameter("raw_value", parameter)
                        constraint = (GreaterThan(0.0) if constraint_type is GreaterThan else Interval(0.0, 1.0)).to(
                            device=device, dtype=dtype
                        )
                        module.register_constraint("raw_value", constraint)
                        module.initialize(raw_value=0.7)
                        self.assertIs(module.raw_value, parameter)
                        self.assertTrue(torch.equal(parameter, torch.full_like(parameter, 0.7)))
                        module.initialize(raw_value=1)
                        self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)))
                        module.register_constraint(
                            "raw_value", GreaterThan(0.8, transform=None).to(device=device, dtype=dtype)
                        )
                        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
                            module.initialize(raw_value=0.7)
                        self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)))

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
