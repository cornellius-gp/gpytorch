#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import NonLazyTensor
from gpytorch.test.base_test_case import BaseTestCase


class TestSqrtInvMatmul(BaseTestCase, unittest.TestCase):
    seed = 0

    def _test(self, matrix_shape, rhs_shape, lhs_shape):
        # Create test matrix, vector
        factor = torch.randn(matrix_shape)
        matrix = (factor.transpose(-1, -2) @ factor).requires_grad_(True)
        rhs = torch.randn(rhs_shape).requires_grad_(True)
        lhs = torch.randn(lhs_shape).requires_grad_(True)

        # Create clones of these factors
        matrix_clone = matrix.clone().detach().requires_grad_(True)
        rhs_clone = rhs.clone().detach().requires_grad_(True)
        lhs_clone = lhs.clone().detach().requires_grad_(True)
        evals, evecs = matrix_clone.symeig(eigenvectors=True)
        matrix_root = evecs @ (evals.sqrt().unsqueeze(-1) * evecs.transpose(-1, -2))

        # Test forward pass
        sqrt_inv_matmul_res, inv_quad_res = NonLazyTensor(matrix).sqrt_inv_matmul(rhs, lhs)
        sqrt_inv_matmul_actual = lhs_clone @ matrix_root.inverse() @ rhs_clone
        inv_quad_actual = ((lhs_clone @ matrix_clone.inverse()) * lhs_clone).sum(dim=-1)

        # Check forward pass
        self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, rtol=1e-4, atol=1e-3)
        self.assertAllClose(inv_quad_res, inv_quad_actual, rtol=1e-4, atol=1e-3)

        # Perform backward pass
        sqrt_inv_matmul_grad = torch.randn_like(sqrt_inv_matmul_res)
        inv_quad_grad = torch.randn_like(inv_quad_res)
        ((sqrt_inv_matmul_res * sqrt_inv_matmul_grad).sum() + (inv_quad_res * inv_quad_grad).sum()).backward()
        ((sqrt_inv_matmul_actual * sqrt_inv_matmul_grad).sum() + (inv_quad_actual * inv_quad_grad).sum()).backward()

        # Check grads
        self.assertAllClose(rhs.grad, rhs_clone.grad, rtol=1e-4, atol=1e-3)
        self.assertAllClose(lhs.grad, lhs_clone.grad, rtol=1e-4, atol=1e-3)
        self.assertAllClose(matrix.grad, matrix_clone.grad, rtol=1e-4, atol=1e-3)

    def test_mat(self):
        return self._test((8, 8), (8, 5), (3, 8))
