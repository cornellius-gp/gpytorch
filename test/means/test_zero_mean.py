#!/usr/bin/env python3

import unittest
from test.means._base_mean_test_case import BaseMeanTestCase
from gpytorch.means import ZeroMean


class TestZeroMean(BaseMeanTestCase, unittest.TestCase):
    def create_mean(self):
        return ZeroMean()
