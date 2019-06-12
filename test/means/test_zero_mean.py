#!/usr/bin/env python3

import unittest

from gpytorch.means import ZeroMean
from gpytorch.test.base_mean_test_case import BaseMeanTestCase


class TestZeroMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return ZeroMean()
