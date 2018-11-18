#!/usr/bin/env python3

import os
import random
import torch
import unittest
from gpytorch.utils import pivoted_cholesky
from test._utils import approx_equal
from gpytorch.kernels import RBFKernel


class TestPivotedCholesky(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.linspace(0, 1, size)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate()
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol.t().matmul(piv_chol)
        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))

    def test_solve_vector(self):
        size = 100
        train_x = torch.linspace(0, 1, size)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate()
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, torch.ones(100))

        rhs_vector = torch.randn(100)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = shifted_covar_matrix.inverse().matmul(rhs_vector)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, torch.ones(100))

        self.assertTrue(approx_equal(approx_solve, real_solve, 2e-4))

    def test_solve(self):
        size = 100
        train_x = torch.linspace(0, 1, size)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate()
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, torch.ones(100))

        rhs_vector = torch.randn(100, 50)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = shifted_covar_matrix.inverse().matmul(rhs_vector)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, torch.ones(100))

        self.assertTrue(approx_equal(approx_solve, real_solve, 2e-4))


class TestPivotedCholeskyBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.cat(
            [torch.linspace(0, 1, size).unsqueeze(0), torch.linspace(0, 0.5, size).unsqueeze(0)], 0
        ).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate()
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol.transpose(1, 2).matmul(piv_chol)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))

    def test_solve(self):
        size = 100
        train_x = torch.cat(
            [torch.linspace(0, 1, size).unsqueeze(0), torch.linspace(0, 0.5, size).unsqueeze(0)], 0
        ).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate()
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, torch.ones(2, 100))

        rhs_vector = torch.randn(2, 100, 5)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = torch.cat(
            [
                shifted_covar_matrix[0].inverse().matmul(rhs_vector[0]).unsqueeze(0),
                shifted_covar_matrix[1].inverse().matmul(rhs_vector[1]).unsqueeze(0),
            ],
            0,
        )
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, torch.ones(2, 100))

        self.assertTrue(approx_equal(approx_solve, real_solve, 2e-4))


class TestPivotedCholeskyMultiBatch(unittest.TestCase):
    def setUp(self):
        if os.getenv("UNLOCK_SEED") is None or os.getenv("UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_pivoted_cholesky(self):
        size = 100
        train_x = torch.cat(
            [
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
            ],
            0,
        ).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate().view(2, 2, 3, size, size)
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        covar_approx = piv_chol.transpose(-1, -2).matmul(piv_chol)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))

    def test_solve(self):
        size = 100
        train_x = torch.cat(
            [
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
                torch.linspace(0, 0.5, size).unsqueeze(0),
                torch.linspace(0, 0.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.25, size).unsqueeze(0),
                torch.linspace(0, 1.5, size).unsqueeze(0),
                torch.linspace(0, 1, size).unsqueeze(0),
            ],
            0,
        ).unsqueeze(-1)
        covar_matrix = RBFKernel()(train_x, train_x).evaluate().view(2, 2, 3, size, size)
        piv_chol = pivoted_cholesky.pivoted_cholesky(covar_matrix, 10)
        woodbury_factor = pivoted_cholesky.woodbury_factor(piv_chol, torch.ones(2, 2, 3, 100))

        rhs_vector = torch.randn(2, 2, 3, 100, 5)
        shifted_covar_matrix = covar_matrix + torch.eye(size)
        real_solve = torch.cat(
            [
                shifted_covar_matrix[0, 0, 0].inverse().matmul(rhs_vector[0, 0, 0]).unsqueeze(0),
                shifted_covar_matrix[0, 0, 1].inverse().matmul(rhs_vector[0, 0, 1]).unsqueeze(0),
                shifted_covar_matrix[0, 0, 2].inverse().matmul(rhs_vector[0, 0, 2]).unsqueeze(0),
                shifted_covar_matrix[0, 1, 0].inverse().matmul(rhs_vector[0, 1, 0]).unsqueeze(0),
                shifted_covar_matrix[0, 1, 1].inverse().matmul(rhs_vector[0, 1, 1]).unsqueeze(0),
                shifted_covar_matrix[0, 1, 2].inverse().matmul(rhs_vector[0, 1, 2]).unsqueeze(0),
                shifted_covar_matrix[1, 0, 0].inverse().matmul(rhs_vector[1, 0, 0]).unsqueeze(0),
                shifted_covar_matrix[1, 0, 1].inverse().matmul(rhs_vector[1, 0, 1]).unsqueeze(0),
                shifted_covar_matrix[1, 0, 2].inverse().matmul(rhs_vector[1, 0, 2]).unsqueeze(0),
                shifted_covar_matrix[1, 1, 0].inverse().matmul(rhs_vector[1, 1, 0]).unsqueeze(0),
                shifted_covar_matrix[1, 1, 1].inverse().matmul(rhs_vector[1, 1, 1]).unsqueeze(0),
                shifted_covar_matrix[1, 1, 2].inverse().matmul(rhs_vector[1, 1, 2]).unsqueeze(0),
            ],
            0,
        ).view_as(rhs_vector)
        approx_solve = pivoted_cholesky.woodbury_solve(rhs_vector, piv_chol, woodbury_factor, torch.ones(2, 3, 100))

        self.assertTrue(approx_equal(approx_solve, real_solve, 2e-4))


if __name__ == "__main__":
    unittest.main()
