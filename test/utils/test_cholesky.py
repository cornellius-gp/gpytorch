#!/usr/bin/env python3

import torch
import unittest
from test._utils import approx_equal
from gpytorch.utils.cholesky import tridiag_batch_potrf, tridiag_batch_potrs


class TestTriDiag(unittest.TestCase):
    def test_potrf(self):
        chol = torch.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 2, 0], [0, 0, 2, 3]], dtype=torch.float).unsqueeze(0)
        trid = chol.matmul(chol.transpose(-1, -2))

        self.assertTrue(torch.equal(chol, tridiag_batch_potrf(trid, upper=False)))

    def test_potrs(self):
        chol = torch.tensor([[1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 2, 0], [0, 0, 2, 3]], dtype=torch.float).unsqueeze(0)

        mat = torch.randn(1, 4, 3)
        self.assertTrue(
            approx_equal(torch.potrs(mat[0], chol[0], upper=False), tridiag_batch_potrs(mat, chol, upper=False)[0])
        )


if __name__ == "__main__":
    unittest.main()
