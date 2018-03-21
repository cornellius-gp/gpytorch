from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch import utils


class TestTeoplitz(unittest.TestCase):

    def test_sym_toeplitz_constructs_tensor_from_vector(self):
        c = torch.Tensor([1, 6, 4, 5])

        res = utils.toeplitz.sym_toeplitz(c)
        actual = torch.Tensor([
            [1, 6, 4, 5],
            [6, 1, 6, 4],
            [4, 6, 1, 6],
            [5, 4, 6, 1],
        ])

        self.assertTrue(torch.equal(res, actual))

    def test_toeplitz_matmul(self):
        col = torch.Tensor([1, 6, 4, 5])
        row = torch.Tensor([1, 2, 1, 1])
        rhs_mat = torch.randn(4, 2)

        # Actual
        lhs_mat = utils.toeplitz.toeplitz(col, row)
        actual = torch.matmul(lhs_mat, rhs_mat)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(col, row, rhs_mat)
        self.assertTrue(utils.approx_equal(res, actual))

    def test_toeplitz_matmul_batch(self):
        cols = torch.Tensor([
            [1, 6, 4, 5],
            [2, 3, 1, 0],
            [1, 2, 3, 1],
        ])
        rows = torch.Tensor([
            [1, 2, 1, 1],
            [2, 0, 0, 1],
            [1, 5, 1, 0],
        ])

        rhs_mats = torch.randn(3, 4, 2)

        # Actual
        lhs_mats = torch.zeros(3, 4, 4)
        for i, (col, row) in enumerate(zip(cols, rows)):
            lhs_mats[i].copy_(utils.toeplitz.toeplitz(col, row))
        actual = torch.matmul(lhs_mats, rhs_mats)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(cols, rows, rhs_mats)
        self.assertTrue(utils.approx_equal(res, actual))

    def test_toeplitz_matmul_batchmat(self):
        col = torch.Tensor([1, 6, 4, 5])
        row = torch.Tensor([1, 2, 1, 1])
        rhs_mat = torch.randn(3, 4, 2)

        # Actual
        lhs_mat = utils.toeplitz.toeplitz(col, row)
        actual = torch.matmul(lhs_mat.unsqueeze(0), rhs_mat)

        # Fast toeplitz
        res = utils.toeplitz.toeplitz_matmul(
            col.unsqueeze(0), row.unsqueeze(0),
            rhs_mat,
        )
        self.assertTrue(utils.approx_equal(res, actual))

    def test_reverse(self):
        input = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        res = torch.Tensor([
            [3, 2, 1],
            [6, 5, 4],
        ])
        self.assertTrue(torch.equal(utils.reverse(input, dim=1), res))

    def test_rcumsum(self):
        input = torch.Tensor([
            [1, 2, 3],
            [4, 5, 6],
        ])
        res = torch.Tensor([
            [6, 5, 3],
            [15, 11, 6],
        ])
        self.assertTrue(torch.equal(utils.rcumsum(input, dim=1), res))


if __name__ == '__main__':
    unittest.main()
