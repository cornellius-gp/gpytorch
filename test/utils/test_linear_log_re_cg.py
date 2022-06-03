import os
import random
import unittest
import torch
from gpytorch.utils.linear_log_re_cg import linear_log_cg_re as linear_cg


class TestLinearCG(unittest.TestCase):
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

    def test_cg(self):
        size = 100
        matrix = torch.randn(size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(size, 50, dtype=torch.float64)
        solves = linear_cg(
            matrix.matmul, rhs=rhs, max_iter=size, tolerance=1.e-1,
            initial_guess=torch.zeros_like(rhs))

        # Check cg
        matrix_chol = torch.linalg.cholesky(matrix)
        actual = torch.cholesky_solve(rhs, matrix_chol)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

    def test_batch_cg(self):
        batch = 5
        size = 100
        matrix = torch.randn(batch, size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(batch, size, 50, dtype=torch.float64)
        solves = linear_cg(
            matrix.matmul, rhs=rhs, max_iter=size, tolerance=1.e-1,
            initial_guess=torch.zeros_like(rhs))

        # Check cg
        matrix_chol = torch.linalg.cholesky(matrix)
        actual = torch.cholesky_solve(rhs, matrix_chol)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
