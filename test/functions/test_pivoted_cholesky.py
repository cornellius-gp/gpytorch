#!/usr/bin/env python3

import unittest

import torch

from gpytorch.functions import pivoted_cholesky
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.permutation import apply_permutation, inverse_permutation


def _ensure_symmetric_grad(grad):
    """
    A gradient-hook hack to ensure that symmetric matrix gradients are symmetric
    """
    res = torch.add(grad, grad.transpose(-1, -2)).mul(0.5)
    return res


class TestPivotedCholesky(BaseTestCase, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(8, 8)
        mat = mat @ mat.transpose(-1, -2)
        return mat

    def test_pivoted_cholesky(self, max_iter=3):
        mat = self._create_mat().detach().requires_grad_(True)
        mat.register_hook(_ensure_symmetric_grad)
        mat_copy = mat.detach().clone().requires_grad_(True)
        mat_copy.register_hook(_ensure_symmetric_grad)

        # Forward (with function)
        res, pivots = pivoted_cholesky(mat, rank=max_iter, return_pivots=True)

        # Forward (manual pivoting, actual Cholesky)
        inverse_pivots = inverse_permutation(pivots)
        # Apply pivoting
        pivoted_mat_copy = apply_permutation(mat_copy, pivots, pivots)
        # Compute Cholesky
        actual_pivoted = psd_safe_cholesky(pivoted_mat_copy)[..., :max_iter]
        # Undo pivoting
        actual = apply_permutation(actual_pivoted, left_permutation=inverse_pivots)

        self.assertAllClose(res, actual)

        # Backward
        grad_output = torch.randn_like(res)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertAllClose(mat.grad, mat_copy.grad)


class TestPivotedCholeskyBatch(TestPivotedCholesky, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(2, 3, 8, 8)
        mat = mat @ mat.transpose(-1, -2)
        return mat


if __name__ == "__main__":
    unittest.main()
