from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import CholLazyTensor
from gpytorch.utils import approx_equal


class TestCholLazyTensor(unittest.TestCase):
    def setUp(self):
        chol = torch.tensor(
            [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]], dtype=torch.float, requires_grad=True
        )
        vecs = torch.randn(5, 2, requires_grad=True)

        self.chol = chol
        self.chol_copy = chol.clone().detach().requires_grad_(True)
        self.actual_mat = self.chol_copy.matmul(self.chol_copy.transpose(-1, -2))
        self.vecs = vecs
        self.vecs_copy = vecs.clone().detach().requires_grad_(True)

    def test_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol).matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol.grad, self.chol_copy.grad))
        self.assertTrue(approx_equal(self.vecs.grad, self.vecs_copy.grad))

    def test_inv_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol).inv_matmul(self.vecs)
        actual = self.actual_mat.inverse().matmul(self.vecs_copy)
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        res_inv_quad, res_log_det = CholLazyTensor(self.chol).inv_quad_log_det(inv_quad_rhs=self.vecs, log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = self.actual_mat.inverse().matmul(self.vecs_copy).mul(self.vecs_copy).sum()
        actual = actual_inv_quad + torch.log(torch.det(self.actual_mat))
        self.assertLess(((res - actual) / actual).abs().item(), 1e-2)

    def test_diag(self):
        res = CholLazyTensor(self.chol).diag()
        actual = self.actual_mat.diag()
        self.assertTrue(approx_equal(res, actual))

    def test_getitem(self):
        res = CholLazyTensor(self.chol)[2:4, -2]
        actual = self.actual_mat[2:4, -2]
        self.assertTrue(approx_equal(res, actual))

    def test_evaluate(self):
        res = CholLazyTensor(self.chol).evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res, actual))


class TestCholLazyTensorBatch(unittest.TestCase):
    def setUp(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
            requires_grad=True,
        )
        vecs = torch.randn(2, 5, 3, requires_grad=True)

        self.chol = chol
        self.chol_copy = chol.clone().detach().requires_grad_(True)
        self.actual_mat = self.chol_copy.matmul(self.chol_copy.transpose(-1, -2))
        self.actual_mat_inv = torch.cat(
            [self.actual_mat[0].inverse().unsqueeze(0), self.actual_mat[1].inverse().unsqueeze(0)], 0
        )

        self.vecs = vecs
        self.vecs_copy = vecs.clone().detach().requires_grad_(True)

    def test_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol).matmul(self.vecs)
        actual = self.actual_mat.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(*self.vecs.size())
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.chol.grad, self.chol_copy.grad))
        self.assertTrue(approx_equal(self.vecs.grad, self.vecs_copy.grad))

    def test_inv_matmul(self):
        # Forward
        res = CholLazyTensor(self.chol).inv_matmul(self.vecs)
        actual = self.actual_mat_inv.matmul(self.vecs_copy)
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_inv_quad_log_det(self):
        # Forward
        res_inv_quad, res_log_det = CholLazyTensor(self.chol).inv_quad_log_det(inv_quad_rhs=self.vecs, log_det=True)
        res = res_inv_quad + res_log_det
        actual_inv_quad = self.actual_mat_inv.matmul(self.vecs_copy).mul(self.vecs_copy).sum(-1).sum(-1)
        actual_log_det = torch.tensor(
            [torch.log(torch.det(self.actual_mat[0])), torch.log(torch.det(self.actual_mat[1]))]
        )

        actual = actual_inv_quad + actual_log_det
        self.assertLess(torch.max((res - actual).abs() / actual.norm()), 1e-2)

    def test_diag(self):
        res = CholLazyTensor(self.chol).diag()
        actual = torch.cat([self.actual_mat[0].diag().unsqueeze(0), self.actual_mat[1].diag().unsqueeze(0)], 0)
        self.assertTrue(approx_equal(res, actual))

    def test_getitem(self):
        res = CholLazyTensor(self.chol)[1, 2:4, -2]
        actual = self.actual_mat[1, 2:4, -2]
        self.assertTrue(approx_equal(res, actual))

    def test_evaluate(self):
        res = CholLazyTensor(self.chol).evaluate()
        actual = self.actual_mat
        self.assertTrue(approx_equal(res, actual))


if __name__ == "__main__":
    unittest.main()
