from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.utils import left_interp, left_t_interp, approx_equal


class TestInterp(unittest.TestCase):

    def setUp(self):
        self.interp_indices = Variable(
            torch.LongTensor([[2, 3], [3, 4], [4, 5]])
        ).repeat(
            3, 1
        )
        self.interp_values = Variable(torch.Tensor([[1, 2], [0.5, 1], [1, 3]])).repeat(
            3, 1
        )
        self.interp_indices_2 = Variable(
            torch.LongTensor([[0, 1], [1, 2], [2, 3]])
        ).repeat(
            3, 1
        )
        self.interp_values_2 = Variable(
            torch.Tensor([[1, 2], [2, 0.5], [1, 3]])
        ).repeat(
            3, 1
        )
        self.batch_interp_indices = torch.cat(
            [self.interp_indices.unsqueeze(0), self.interp_indices_2.unsqueeze(0)], 0
        )
        self.batch_interp_values = torch.cat(
            [self.interp_values.unsqueeze(0), self.interp_values_2.unsqueeze(0)], 0
        )
        self.interp_matrix = torch.Tensor(
            [
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 0, 0.5, 1, 0],
                [0, 0, 0, 0, 1, 3],
            ]
        )

        self.batch_interp_matrix = torch.Tensor(
            [
                [
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                    [0, 0, 1, 2, 0, 0],
                    [0, 0, 0, 0.5, 1, 0],
                    [0, 0, 0, 0, 1, 3],
                ],
                [
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                    [1, 2, 0, 0, 0, 0],
                    [0, 2, 0.5, 0, 0, 0],
                    [0, 0, 1, 3, 0, 0],
                ],
            ]
        )

    def test_left_interp_on_a_vector(self):
        vector = torch.randn(6)

        res = left_interp(
            self.interp_indices, self.interp_values, Variable(vector)
        ).data
        actual = torch.matmul(self.interp_matrix, vector)
        self.assertTrue(approx_equal(res, actual))

    def test_left_t_interp_on_a_vector(self):
        vector = torch.randn(9)

        res = left_t_interp(
            self.interp_indices, self.interp_values, Variable(vector), 6
        ).data
        actual = torch.matmul(self.interp_matrix.transpose(-1, -2), vector)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_vector(self):
        vector = torch.randn(6)

        actual = torch.matmul(
            self.batch_interp_matrix, vector.unsqueeze(-1).unsqueeze(0)
        ).squeeze(
            -1
        )
        res = left_interp(
            self.batch_interp_indices, self.batch_interp_values, Variable(vector)
        ).data
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_vector(self):
        vector = torch.randn(9)

        actual = torch.matmul(
            self.batch_interp_matrix.transpose(-1, -2),
            vector.unsqueeze(-1).unsqueeze(0),
        ).squeeze(
            -1
        )
        res = left_t_interp(
            self.batch_interp_indices, self.batch_interp_values, Variable(vector), 6
        ).data
        self.assertTrue(approx_equal(res, actual))

    def test_left_interp_on_a_matrix(self):
        matrix = torch.randn(6, 3)

        res = left_interp(
            self.interp_indices, self.interp_values, Variable(matrix)
        ).data
        actual = torch.matmul(self.interp_matrix, matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_left_t_interp_on_a_matrix(self):
        matrix = torch.randn(9, 3)

        res = left_t_interp(
            self.interp_indices, self.interp_values, Variable(matrix), 6
        ).data
        actual = torch.matmul(self.interp_matrix.transpose(-1, -2), matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_matrix(self):
        batch_matrix = torch.randn(6, 3)

        res = left_interp(
            self.batch_interp_indices, self.batch_interp_values, Variable(batch_matrix)
        ).data
        actual = torch.matmul(self.batch_interp_matrix, batch_matrix.unsqueeze(0))
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_matrix(self):
        batch_matrix = torch.randn(9, 3)

        res = left_t_interp(
            self.batch_interp_indices,
            self.batch_interp_values,
            Variable(batch_matrix),
            6,
        ).data
        actual = torch.matmul(
            self.batch_interp_matrix.transpose(-1, -2), batch_matrix.unsqueeze(0)
        )
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_interp_on_a_batch_matrix(self):
        batch_matrix = torch.randn(2, 6, 3)

        res = left_interp(
            self.batch_interp_indices, self.batch_interp_values, Variable(batch_matrix)
        ).data
        actual = torch.matmul(self.batch_interp_matrix, batch_matrix)
        self.assertTrue(approx_equal(res, actual))

    def test_batch_left_t_interp_on_a_batch_matrix(self):
        batch_matrix = torch.randn(2, 9, 3)

        res = left_t_interp(
            self.batch_interp_indices,
            self.batch_interp_values,
            Variable(batch_matrix),
            6,
        ).data
        actual = torch.matmul(self.batch_interp_matrix.transpose(-1, -2), batch_matrix)
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
