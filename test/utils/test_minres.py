#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.base_test_case import BaseTestCase
from gpytorch.utils.minres import minres


class TestMinres(BaseTestCase, unittest.TestCase):
    seed = 0

    def _test_minres(self, rhs_shape, shifts=None, matrix_batch_shape=torch.Size([])):
        size = rhs_shape[-2] if len(rhs_shape) > 1 else rhs_shape[-1]
        rhs = torch.randn(rhs_shape, dtype=torch.float64)

        matrix = torch.randn(*matrix_batch_shape, size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(size, dtype=torch.float64).mul_(1e-1))

        # Compute solves with minres
        if shifts is not None:
            shifts = shifts.type_as(rhs)

        with gpytorch.settings.minres_tolerance(1e-6):
            solves = minres(matrix, rhs=rhs, value=-1, shifts=shifts)

        # Make sure that we're not getting weird batch dim effects
        while matrix.dim() < len(rhs_shape):
            matrix = matrix.unsqueeze(0)

        # Maybe add shifts
        if shifts is not None:
            matrix = matrix - torch.mul(
                torch.eye(size, dtype=torch.float64), shifts.view(*shifts.shape, *[1 for _ in matrix.shape])
            )

        # Compute solves exactly
        actual, _ = torch.solve(rhs.unsqueeze(-1) if rhs.dim() == 1 else rhs, -matrix)
        if rhs.dim() == 1:
            actual = actual.squeeze(-1)

        self.assertAllClose(solves, actual, atol=1e-3, rtol=1e-4)

    def test_minres_vec(self):
        return self._test_minres(torch.Size([20]))

    def test_minres_vec_multiple_shifts(self):
        shifts = torch.tensor([0.0, 1.0, 2.0])
        return self._test_minres(torch.Size([5]), shifts=shifts)

    def test_minres_mat(self):
        self._test_minres(torch.Size([20, 5]))
        self._test_minres(torch.Size([3, 20, 5]))
        self._test_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]))
        return self._test_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]))

    def test_minres_mat_multiple_shifts(self):
        shifts = torch.tensor([0.0, 1.0, 2.0])
        self._test_minres(torch.Size([20, 5]), shifts=shifts)
        self._test_minres(torch.Size([3, 20, 5]), shifts=shifts)
        self._test_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)
        return self._test_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)
