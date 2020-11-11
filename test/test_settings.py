#!/usr/bin/env python3

import unittest

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
