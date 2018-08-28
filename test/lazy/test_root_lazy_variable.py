from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import RootLazyVariable
from gpytorch.utils import approx_equal


class TestRootLazyVariable(unittest.TestCase):
    def test_matmul(self):
        root = torch.randn(5, 3, requires_grad=True)
        covar = RootLazyVariable(root)
        mat = torch.eye(5)
        res = covar.matmul(mat)

        root_clone = root.clone().detach()
        root_clone.requires_grad = True
        mat_clone = mat.clone().detach()
        mat_clone.requires_grad = True
        actual = root_clone.matmul(root_clone.transpose(-1, -2)).matmul(mat_clone)

        self.assertTrue(approx_equal(res, actual))

        gradient = torch.randn(5, 5)
        actual.backward(gradient=gradient)
        res.backward(gradient=gradient)

        self.assertTrue(approx_equal(root.grad, root_clone.grad))

    def test_diag(self):
        root = torch.randn(5, 3)
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual.diag(), res.diag()))

    def test_batch_diag(self):
        root = torch.randn(4, 5, 3)
        actual = root.matmul(root.transpose(-1, -2))
        actual_diag = torch.cat(
            [
                actual[0].diag().unsqueeze(0),
                actual[1].diag().unsqueeze(0),
                actual[2].diag().unsqueeze(0),
                actual[3].diag().unsqueeze(0),
            ]
        )

        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual_diag, res.diag()))

    def test_batch_get_indices(self):
        root = torch.randn(2, 5, 1)
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)

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
        root = torch.randn(5, 3)
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)

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
        root = torch.randn(5, 3)
        actual = root.matmul(root.transpose(-1, -2))
        res = RootLazyVariable(root)
        self.assertTrue(approx_equal(actual, res.evaluate()))


if __name__ == "__main__":
    unittest.main()
