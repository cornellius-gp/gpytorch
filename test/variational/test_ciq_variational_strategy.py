#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestCiqVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.CiqVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 4)  # One for each forward pass, one for each backward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(cholesky_mock.called)
        self.assertEqual(ciq_mock.call_count, 2)  # One for each evaluation call


class TestMeanFieldCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestDeltaCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestNgdCiqVariationalGP(TestCiqVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.NaturalVariationalDistribution


if __name__ == "__main__":
    unittest.main()
