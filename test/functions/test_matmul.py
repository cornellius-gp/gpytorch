from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from torch.autograd import Variable
from gpytorch.lazy import NonLazyTensor
from gpytorch.utils import approx_equal


class TestMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = torch.Tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]])
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)

        self.mat_var = Variable(mat, requires_grad=True)
        self.mat_var_clone = Variable(mat, requires_grad=True)
        self.vec_var = Variable(vec, requires_grad=True)
        self.vec_var_clone = Variable(vec, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_matmul_vec(self):
        # Forward
        res = NonLazyTensor(self.mat_var).matmul(self.vec_var)
        actual = self.mat_var_clone.matmul(self.vec_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(3).normal_()
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vec_var_clone.grad.data, self.vec_var.grad.data))

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mat_var).matmul(self.vecs_var)
        actual = self.mat_var_clone.matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(3, 4).normal_()
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_var_clone.grad.data, self.mat_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


class TestMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = torch.Tensor([[[3, -1, 0], [-1, 3, 0], [0, 0, 3]], [[10, -2, 1], [-2, 10, 0], [1, 0, 10]]])
        vecs = torch.randn(2, 3, 4)

        self.mats_var = Variable(mats, requires_grad=True)
        self.mats_var_clone = Variable(mats, requires_grad=True)
        self.vecs_var = Variable(vecs, requires_grad=True)
        self.vecs_var_clone = Variable(vecs, requires_grad=True)

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mats_var).matmul(self.vecs_var)
        actual = self.mats_var_clone.matmul(self.vecs_var_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.Tensor(2, 3, 4).normal_()
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_var_clone.grad.data, self.mats_var.grad.data))
        self.assertTrue(approx_equal(self.vecs_var_clone.grad.data, self.vecs_var.grad.data))


if __name__ == "__main__":
    unittest.main()
