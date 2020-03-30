#!/usr/bin/env python3

import unittest

import torch

from gpytorch import settings
from gpytorch.lazy import NonLazyTensor
from gpytorch.test.base_test_case import BaseTestCase


class TestSqrtInvMatmul(BaseTestCase, unittest.TestCase):
    seed = 0

    def _test(self, matrix_shape, rhs_shape, lhs_shape):
        with settings.record_ciq_stats(), settings.num_contour_quadrature(31), settings.minres_tolerance(1e-10):
            # Create test matrix, vector
            factor = torch.randn(matrix_shape)
            matrix = factor.transpose(-1, -2) @ factor
            matrix = matrix.div_(matrix.max())
            matrix += torch.eye(matrix.size(-1)).mul_(1e-2)
            matrix = matrix.double()
            matrix.requires_grad_(True)
            lhs = torch.randn(lhs_shape, dtype=torch.double).requires_grad_(True)
            rhs = torch.randn(rhs_shape, dtype=lhs.dtype).detach().requires_grad_(True)

            # Create clones of these factors
            matrix_clone = matrix.clone().detach().requires_grad_(True)
            rhs_clone = rhs.clone().detach().requires_grad_(True)
            lhs_clone = lhs.clone().detach().requires_grad_(True)
            evals, evecs = matrix_clone.double().symeig(eigenvectors=True)
            evals = evals.to(matrix_clone.dtype)
            evecs = evecs.to(matrix_clone.dtype)
            self.assertAllClose(
                evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2)), matrix_clone, rtol=1e-6, atol=1e-6
            )
            matrix_root = evecs @ (evals.sqrt().unsqueeze(-1) * evecs.transpose(-1, -2))
            matrix_chol = matrix_clone.cholesky()

            # Test forward pass
            sqrt_inv_matmul_res, inv_quad_res = NonLazyTensor(matrix).sqrt_inv_matmul(rhs, lhs)
            sqrt_inv_matmul_actual = lhs_clone @ matrix_root.inverse() @ rhs_clone
            # inv_quad_actual = (lhs_clone @ matrix_root.inverse()).pow(2).sum(dim=-1)
            inv_quad_actual = (
                torch.triangular_solve(lhs_clone.transpose(-1, -2), matrix_chol, upper=False)[0].pow(2).sum(dim=-2)
            )

            # Check forward pass
            self.assertAllClose(inv_quad_res, inv_quad_actual, rtol=1e-5, atol=1e-4)
            self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, rtol=1e-5, atol=1e-4)

        # Perform backward pass
        sqrt_inv_matmul_grad = torch.randn_like(sqrt_inv_matmul_res)
        inv_quad_grad = torch.randn_like(inv_quad_res)
        ((sqrt_inv_matmul_res * sqrt_inv_matmul_grad).sum() + (inv_quad_res * inv_quad_grad).sum()).backward()
        ((sqrt_inv_matmul_actual * sqrt_inv_matmul_grad).sum() + (inv_quad_actual * inv_quad_grad).sum()).backward()

        # Check grads
        self.assertAllClose(rhs.grad, rhs_clone.grad, rtol=1e-5, atol=1e-4)
        self.assertAllClose(lhs.grad, lhs_clone.grad, rtol=1e-5, atol=1e-4)
        self.assertAllClose(matrix.grad, matrix_clone.grad, rtol=1e-4, atol=1e-3)

    def _test_no_lhs(self, matrix_shape, rhs_shape):
        with settings.record_ciq_stats(), settings.num_contour_quadrature(31), settings.minres_tolerance(1e-10):
            # Create test matrix, vector
            factor = torch.randn(matrix_shape)
            matrix = factor.transpose(-1, -2) @ factor
            matrix = matrix.div_(matrix.max())
            matrix += torch.eye(matrix.size(-1)).mul_(1e-2)
            matrix = matrix.double()
            matrix.requires_grad_(True)
            rhs = torch.randn(rhs_shape, dtype=matrix.dtype).detach().requires_grad_(True)

            # Create clones of these factors
            matrix_clone = matrix.clone().detach().requires_grad_(True)
            rhs_clone = rhs.clone().detach().requires_grad_(True)
            evals, evecs = matrix_clone.double().symeig(eigenvectors=True)
            evals = evals.to(matrix_clone.dtype)
            evecs = evecs.to(matrix_clone.dtype)
            self.assertAllClose(
                evecs @ (evals.unsqueeze(-1) * evecs.transpose(-1, -2)), matrix_clone, rtol=1e-6, atol=1e-6
            )
            matrix_root = evecs @ (evals.sqrt().unsqueeze(-1) * evecs.transpose(-1, -2))

            # Test forward pass
            sqrt_inv_matmul_res = NonLazyTensor(matrix).sqrt_inv_matmul(rhs)
            sqrt_inv_matmul_actual = matrix_root.inverse() @ rhs_clone

            # Check forward pass
            self.assertAllClose(sqrt_inv_matmul_res, sqrt_inv_matmul_actual, rtol=1e-5, atol=1e-4)

            # Perform backward pass
            sqrt_inv_matmul_grad = torch.randn_like(sqrt_inv_matmul_res)
            ((sqrt_inv_matmul_res * sqrt_inv_matmul_grad).sum()).backward()
            ((sqrt_inv_matmul_actual * sqrt_inv_matmul_grad).sum()).backward()

            # Check grads
            self.assertAllClose(rhs.grad, rhs_clone.grad, rtol=1e-5, atol=1e-4)
            self.assertAllClose(matrix.grad, matrix_clone.grad, rtol=1e-4, atol=1e-3)

    def test_mat(self):
        return self._test((128, 128), (128, 5), (1, 128))

    def test_mat_no_lhs(self):
        return self._test_no_lhs((128, 128), (128, 5))
