from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import NonLazyTensor
from gpytorch.utils import approx_equal


class TestMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = [[3, -1, 0], [-1, 3, 0], [0, 0, 3]]
        vec = torch.randn(3, requires_grad=True)
        vecs = torch.randn(3, 4, requires_grad=True)

        self.mat_var = torch.tensor(mat, dtype=torch.float, requires_grad=True)
        self.mat_var_copy = self.mat_var.clone().detach().requires_grad_(True)
        self.vec_var = vec
        self.vec_var_copy = vec.clone().detach().requires_grad_(True)
        self.vecs_var = vecs
        self.vecs_var_copy = vecs.clone().detach().requires_grad_(True)

    def test_matmul_vec(self):
        # Forward
        res = NonLazyTensor(self.mat_var).matmul(self.vec_var)
        actual = self.mat_var_copy.matmul(self.vec_var_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_copy.grad, self.mat_var.grad))
        self.assertTrue(approx_equal(self.vec_var_copy.grad, self.vec_var.grad))

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mat_var).matmul(self.vecs_var)
        actual = self.mat_var_copy.matmul(self.vecs_var_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_copy.grad, self.mat_var.grad))
        self.assertTrue(approx_equal(self.vecs_var_copy.grad, self.vecs_var.grad))


class TestMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = [[[3, -1, 0], [-1, 3, 0], [0, 0, 3]], [[10, -2, 1], [-2, 10, 0], [1, 0, 10]]]
        vecs = torch.randn(2, 3, 4, requires_grad=True)

        self.mats_var = torch.tensor(mats, dtype=torch.float, requires_grad=True)
        self.mats_var_clone = self.mats_var.clone().detach().requires_grad_(True)
        self.vecs_var = vecs
        self.vecs_var_copy = vecs.clone().detach().requires_grad_(True)

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mats_var).matmul(self.vecs_var)
        actual = self.mats_var_clone.matmul(self.vecs_var_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(2, 3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad, self.mats_var.grad))
        self.assertTrue(approx_equal(self.vecs_var_copy.grad, self.vecs_var.grad))


if __name__ == "__main__":
    unittest.main()
