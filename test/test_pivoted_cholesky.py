#!/usr/bin/env python3

import unittest

import linear_operator
import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase


class TestPivotedCholesky(BaseTestCase, unittest.TestCase):
    def _psd_matrix(self, n=20):
        # A smooth kernel matrix: the spectrum decays quickly, so a loose
        # error_tol stops the factorization before it reaches `rank` columns.
        x = torch.linspace(0, 1, n).unsqueeze(-1)
        return torch.exp(-((x - x.transpose(-1, -2)) ** 2) / 0.5) + 1e-6 * torch.eye(n)

    def test_error_tol_is_respected(self):
        mat = self._psd_matrix()
        rank = 10

        without_tol = gpytorch.pivoted_cholesky(mat, rank=rank)
        with_tol = gpytorch.pivoted_cholesky(mat, rank=rank, error_tol=1e-1)

        # error_tol is a stopping criterion, so it has to produce a strictly
        # smaller factor than the same call without it.
        self.assertLess(with_tol.size(-1), without_tol.size(-1))
        self.assertAllClose(with_tol, linear_operator.pivoted_cholesky(mat, rank=rank, error_tol=1e-1))


if __name__ == "__main__":
    unittest.main()
