#!/usr/bin/env python3

import unittest

import torch

from gpytorch import utils
from gpytorch.test.utils import approx_equal


class TestToeplitz(unittest.TestCase):
    def test_sym_toeplitz_constructs_tensor_from_vector(self):
        c = torch.tensor([1, 6, 4, 5], dtype=torch.float)

        res = utils.toeplitz.sym_toeplitz(c)
        actual = torch.tensor([[1, 6, 4, 5], [6, 1, 6, 4], [4, 6, 1, 6], [5, 4, 6, 1]], dtype=torch.float)

        self.assertTrue(torch.equal(res, actual))

    def test_toeplitz_matmul(self):
        col = torch.tensor([1, 6, 4, 5], dtype=torch.float)
        row = torch.tensor([1, 2, 1, 1], dtype=torch.float)
        rhs_mat = torch.randn(4, 2)

        # Actual
        lhs_mat = utils.toeplitz.toeplitz(col, row)
        actual = torch.matmul(lhs_mat, rhs_mat)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(col, row, rhs_mat)
        self.assertTrue(approx_equal(res, actual))

    def test_toeplitz_matmul_batch(self):
        cols = torch.tensor([[1, 6, 4, 5], [2, 3, 1, 0], [1, 2, 3, 1]], dtype=torch.float)
        rows = torch.tensor([[1, 2, 1, 1], [2, 0, 0, 1], [1, 5, 1, 0]], dtype=torch.float)

        rhs_mats = torch.randn(3, 4, 2)

        # Actual
        lhs_mats = torch.zeros(3, 4, 4)
        for i, (col, row) in enumerate(zip(cols, rows)):
            lhs_mats[i].copy_(utils.toeplitz.toeplitz(col, row))
        actual = torch.matmul(lhs_mats, rhs_mats)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(cols, rows, rhs_mats)
        self.assertTrue(approx_equal(res, actual))

    def test_toeplitz_matmul_batchmat(self):
        col = torch.tensor([1, 6, 4, 5], dtype=torch.float)
        row = torch.tensor([1, 2, 1, 1], dtype=torch.float)
        rhs_mat = torch.randn(3, 4, 2)

        # Actual
        lhs_mat = utils.toeplitz.toeplitz(col, row)
        actual = torch.matmul(lhs_mat.unsqueeze(0), rhs_mat)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(col.unsqueeze(0), row.unsqueeze(0), rhs_mat)
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
