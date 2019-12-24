#!/usr/bin/env python3

import math
import os
import random
import unittest

import torch

from gpytorch import settings
from gpytorch.kernels import RBFKernel
from gpytorch.test.utils import approx_equal
from gpytorch.utils import pivoted_cholesky


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
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)
        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))

    def test_solve_qr(self, dtype=torch.float64, tol=1e-8):
        size = 50
        X = torch.rand((size, 2)).to(dtype=dtype)
        y = torch.sin(torch.sum(X, 1)).unsqueeze(-1).to(dtype=dtype)
        with settings.min_preconditioning_size(0):
            noise = torch.DoubleTensor(size).uniform_(math.log(1e-3), math.log(1e-1)).exp_().to(dtype=dtype)
            lazy_tsr = RBFKernel().to(dtype=dtype)(X).evaluate_kernel().add_diag(noise)
            precondition_qr, _, logdet_qr = lazy_tsr._preconditioner()

            F = lazy_tsr._piv_chol_self
            M = noise.diag() + F.matmul(F.t())

        x_exact = torch.solve(y, M)[0]
        x_qr = precondition_qr(y)

        self.assertTrue(approx_equal(x_exact, x_qr, tol))

        logdet = 2 * torch.cholesky(M).diag().log().sum(-1)
        self.assertTrue(approx_equal(logdet, logdet_qr, tol))

    def test_solve_qr_constant_noise(self, dtype=torch.float64, tol=1e-8):
        size = 50
        X = torch.rand((size, 2)).to(dtype=dtype)
        y = torch.sin(torch.sum(X, 1)).unsqueeze(-1).to(dtype=dtype)

        with settings.min_preconditioning_size(0):
            noise = 1e-2 * torch.ones(size, dtype=dtype)
            lazy_tsr = RBFKernel().to(dtype=dtype)(X).evaluate_kernel().add_diag(noise)
            precondition_qr, _, logdet_qr = lazy_tsr._preconditioner()

            F = lazy_tsr._piv_chol_self
        M = noise.diag() + F.matmul(F.t())

        x_exact = torch.solve(y, M)[0]
        x_qr = precondition_qr(y)

        self.assertTrue(approx_equal(x_exact, x_qr, tol))

        logdet = 2 * torch.cholesky(M).diag().log().sum(-1)
        self.assertTrue(approx_equal(logdet, logdet_qr, tol))

    def test_solve_qr_float32(self):
        self.test_solve_qr(dtype=torch.float32, tol=1e-2)

    def test_solve_qr_constant_noise_float32(self):
        self.test_solve_qr_constant_noise(dtype=torch.float32, tol=1e-3)


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
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))


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
        covar_approx = piv_chol @ piv_chol.transpose(-1, -2)

        self.assertTrue(approx_equal(covar_approx, covar_matrix, 2e-4))


if __name__ == "__main__":
    unittest.main()
