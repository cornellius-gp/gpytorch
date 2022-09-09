#!/usr/bin/env python3

import unittest

import torch

from gpytorch import settings
from gpytorch.test.base_test_case import BaseTestCase


class TestSettings(BaseTestCase, unittest.TestCase):
    def test_feature_flag(self):
        self.assertTrue(settings.fast_pred_var.is_default())
        self.assertFalse(settings.fast_pred_var.on())
        with settings.fast_pred_var():
            self.assertFalse(settings.fast_pred_var.is_default())
            self.assertTrue(settings.fast_pred_var.on())
        with settings.fast_pred_var(False):
            self.assertFalse(settings.fast_pred_var.is_default())
            self.assertFalse(settings.fast_pred_var.on())

    def test_dtype_value_context(self):
        # test custom settings
        x = torch.zeros(1, dtype=torch.float)
        with settings.min_fixed_noise(float_value=0.1, double_value=0.2, half_value=0.3):
            self.assertEqual(settings.min_fixed_noise.value(x), 0.1)
            self.assertEqual(settings.min_fixed_noise.value(x.double()), 0.2)
            self.assertEqual(settings.min_fixed_noise.value(x.half()), 0.3)
        # test defaults are restored
        self.assertEqual(
            settings.min_fixed_noise.value(x),
            settings.min_fixed_noise._global_float_value,
        )
        self.assertEqual(
            settings.min_fixed_noise.value(x.double()),
            settings.min_fixed_noise._global_double_value,
        )
        self.assertEqual(
            settings.min_fixed_noise.value(x.half()),
            settings.min_fixed_noise._global_half_value,
        )
        # test setting one dtype
        with settings.min_fixed_noise(double_value=0.2):
            self.assertEqual(settings.min_fixed_noise.value(x.double()), 0.2)
            self.assertEqual(
                settings.min_fixed_noise.value(x),
                settings.min_fixed_noise._global_float_value,
            )
            self.assertEqual(
                settings.min_fixed_noise.value(x.half()),
                settings.min_fixed_noise._global_half_value,
            )
