#!/usr/bin/env python3

import unittest
import torch
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.utils.minres import minres
from gpytorch.lazy import NonLazyTensor


class TestSqrtInvMatmul(BaseTestCase, unittest.TestCase):
    seed = 0

    def _test(self, matrix_shape, rhs_shape):
        # Create test matrix, vector
        factor = torch.randn(matrix_shape)
        matrix = (factor.transpose(-1, -2) @ factor).requires_grad_(True)
        bar = torch.randn(matrix_shape)
        rhs = torch.eye(matrix_shape[-2])

        # Create clones of these factors
        matrix_clone = matrix.clone().detach().requires_grad_(True)
        rhs_clone = rhs.clone().detach().requires_grad_(True)
        evals, evecs = matrix_clone.symeig(eigenvectors=True)
        matrix_root = evecs @ (evals.sqrt().unsqueeze(-1) * evecs.transpose(-1, -2))

        # Test forward pass
        res = NonLazyTensor(matrix).sqrt_inv_matmul(rhs)
        actual = matrix_root.inverse() @ rhs_clone
        self.assertAllClose(res, actual, atol=1e-3, rtol=1e-4)

        # Test backward pass
        # grad = torch.rand_like(res)
        grad = bar.clone().detach()
        res.backward(gradient=grad)
        actual.backward(gradient=grad)
        self.assertAllClose(
            (matrix.grad + matrix.grad.transpose(-1, -2)).div(2.),
            (matrix_clone.grad + matrix_clone.grad.transpose(-1, -2)).div(2.),
            atol=1e-3, rtol=1e-4
        )
        self.assertAllClose(rhs.grad, rhs_clone.grad, atol=1e-3, rtol=1e-4)

    # def test_vec(self):
        # return self._test((5, 5), (5,))

    def test_mat(self):
        return self._test((5, 5), (5, 3))
