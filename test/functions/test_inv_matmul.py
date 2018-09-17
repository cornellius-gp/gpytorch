from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import NonLazyTensor
from gpytorch.utils import approx_equal


class TestInvMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = [[3, -1, 0], [-1, 3, 0], [0, 0, 3]]
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = torch.tensor(mat, dtype=torch.float, requires_grad=True)
        self.mat_var_clone = torch.tensor(mat, dtype=torch.float, requires_grad=True)
        self.vec_var = torch.tensor(vec, requires_grad=True)
        self.vec_var_clone = torch.tensor(vec, requires_grad=True)
        self.vecs_var = torch.tensor(vecs, requires_grad=True)
        self.vecs_var_clone = torch.tensor(vecs, requires_grad=True)

    def test_inv_matmul_vec(self):
        # Forward
        res = NonLazyTensor(self.mat_var).inv_matmul(self.vec_var)
        actual = self.mat_var_clone.inverse().matmul(self.vec_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad))
        self.assertTrue(approx_equal(self.vec_var_clone.grad, self.vec_var.grad))

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mat_var).inv_matmul(self.vecs_var)
        actual = self.mat_var_clone.inverse().matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad, self.mat_var.grad))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))


class TestInvMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = [[[3, -1, 0], [-1, 3, 0], [0, 0, 3]], [[10, -2, 1], [-2, 10, 0], [1, 0, 10]]]
        vecs = torch.randn(2, 3, 4)

        self.mats_var = torch.tensor(mats, dtype=torch.float, requires_grad=True)
        self.mats_var_clone = torch.tensor(mats, dtype=torch.float, requires_grad=True)
        self.vecs_var = torch.tensor(vecs, requires_grad=True)
        self.vecs_var_clone = torch.tensor(vecs, requires_grad=True)

    def test_inv_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mats_var).inv_matmul(self.vecs_var)
        actual = torch.cat(
            [self.mats_var_clone[0].inverse().unsqueeze(0), self.mats_var_clone[1].inverse().unsqueeze(0)]
        ).matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(2, 3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad, self.mats_var.grad))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad, self.vecs_var.grad))


if __name__ == "__main__":
    unittest.main()
