from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import MatmulLazyTensor
from gpytorch.utils import approx_equal


class TestMatmulLazyTensor(unittest.TestCase):
    def test_matmul(self):
        lhs = torch.randn(5, 3, requires_grad=True)
        rhs = torch.randn(3, 4, requires_grad=True)
        covar = MatmulLazyTensor(lhs, rhs)
        mat = torch.randn(4, 10)
        res = covar.matmul(mat)

        lhs_clone = lhs.clone().detach()
        rhs_clone = rhs.clone().detach()
        mat_clone = mat.clone().detach()
        lhs_clone.requires_grad = True
        rhs_clone.requires_grad = True
        mat_clone.requires_grad = True
        actual = lhs_clone.matmul(rhs_clone).matmul(mat_clone)

        self.assertTrue(approx_equal(res, actual))

        actual.sum().backward()

        res.sum().backward()
        self.assertTrue(approx_equal(lhs.grad, lhs_clone.grad))
        self.assertTrue(approx_equal(rhs.grad, rhs_clone.grad))

    def test_diag(self):
        lhs = torch.randn(5, 3)
        rhs = torch.randn(3, 5)
        actual = lhs.matmul(rhs)
        res = MatmulLazyTensor(lhs, rhs)
        self.assertTrue(approx_equal(actual.diag(), res.diag()))

    def test_batch_diag(self):
        lhs = torch.randn(4, 5, 3)
        rhs = torch.randn(4, 3, 5)
        actual = lhs.matmul(rhs)
        actual_diag = torch.cat(
            [
                actual[0].diag().unsqueeze(0),
                actual[1].diag().unsqueeze(0),
                actual[2].diag().unsqueeze(0),
                actual[3].diag().unsqueeze(0),
            ]
        )

        res = MatmulLazyTensor(lhs, rhs)
        self.assertTrue(approx_equal(actual_diag, res.diag()))

    def test_batch_get_indices(self):
        lhs = torch.randn(2, 5, 1)
        rhs = torch.randn(2, 1, 5)
        actual = lhs.matmul(rhs)
        res = MatmulLazyTensor(lhs, rhs)

        batch_indices = torch.LongTensor([0, 1, 0, 1])
        left_indices = torch.LongTensor([1, 2, 4, 0])
        right_indices = torch.LongTensor([0, 1, 3, 2])

        self.assertTrue(
            approx_equal(
                actual[batch_indices, left_indices, right_indices],
                res._batch_get_indices(batch_indices, left_indices, right_indices),
            )
        )

        batch_indices = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        left_indices = torch.LongTensor([1, 2, 4, 0, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0, 4, 4, 4, 4])
        right_indices = torch.LongTensor([0, 1, 3, 2, 3, 4, 2, 2, 1, 1, 2, 1, 2, 4, 4, 3, 3, 0])

        self.assertTrue(
            approx_equal(
                actual[batch_indices, left_indices, right_indices],
                res._batch_get_indices(batch_indices, left_indices, right_indices),
            )
        )

    def test_get_indices(self):
        lhs = torch.randn(5, 1)
        rhs = torch.randn(1, 5)
        actual = lhs.matmul(rhs)
        res = MatmulLazyTensor(lhs, rhs)

        left_indices = torch.LongTensor([1, 2, 4, 0])
        right_indices = torch.LongTensor([0, 1, 3, 2])

        self.assertTrue(
            approx_equal(actual[left_indices, right_indices], res._get_indices(left_indices, right_indices))
        )

        left_indices = torch.LongTensor([1, 2, 4, 0, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0, 4, 4, 4, 4])
        right_indices = torch.LongTensor([0, 1, 3, 2, 3, 4, 2, 2, 1, 1, 2, 1, 2, 4, 4, 3, 3, 0])

        self.assertTrue(
            approx_equal(actual[left_indices, right_indices], res._get_indices(left_indices, right_indices))
        )

    def test_evaluate(self):
        lhs = torch.randn(5, 3)
        rhs = torch.randn(3, 5)
        actual = lhs.matmul(rhs)
        res = MatmulLazyTensor(lhs, rhs)
        self.assertTrue(approx_equal(actual, res.evaluate()))

    def test_transpose(self):
        lhs = torch.randn(5, 3)
        rhs = torch.randn(3, 5)
        actual = lhs.matmul(rhs)
        res = MatmulLazyTensor(lhs, rhs)
        self.assertTrue(approx_equal(actual.t(), res.t().evaluate()))


if __name__ == "__main__":
    unittest.main()
