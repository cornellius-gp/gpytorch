#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import NonLazyTensor
from gpytorch.test.utils import approx_equal


class TestMatmulNonBatch(unittest.TestCase):
    def setUp(self):
        mat = torch.tensor([[3, -1, 0], [-1, 3, 0], [0, 0, 3]], dtype=torch.float)
        vec = torch.randn(3)
        vecs = torch.randn(3, 4)
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_copy = mat.detach().clone().requires_grad_(True)
        self.vec = vec.detach().clone().requires_grad_(True)
        self.vec_copy = vec.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_matmul_vec(self):
        # Forward
        res = NonLazyTensor(self.mat).matmul(self.vec)
        actual = self.mat_copy.matmul(self.vec_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_copy.grad, self.mat.grad))
        self.assertTrue(approx_equal(self.vec_copy.grad, self.vec.grad))

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mat).matmul(self.vecs)
        actual = self.mat_copy.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mat_copy.grad, self.mat.grad))
        self.assertTrue(approx_equal(self.vecs_copy.grad, self.vecs.grad))


class TestMatmulBatch(unittest.TestCase):
    def setUp(self):
        mats = torch.randn(2, 5, 3)
        vecs = torch.randn(2, 3, 4)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_copy = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mats).matmul(self.vecs)
        actual = self.mats_copy.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(2, 5, 4)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_copy.grad, self.mats.grad))
        self.assertTrue(approx_equal(self.vecs_copy.grad, self.vecs.grad))


class TestMatmulMultiBatch(unittest.TestCase):
    def setUp(self):
        mats = torch.randn(3, 4, 5, 6)
        vecs = torch.randn(3, 4, 6, 2)
        self.mats = mats.detach().clone().requires_grad_(True)
        self.mats_copy = mats.detach().clone().requires_grad_(True)
        self.vecs = vecs.detach().clone().requires_grad_(True)
        self.vecs_copy = vecs.detach().clone().requires_grad_(True)

    def test_matmul_multiple_vecs(self):
        # Forward
        res = NonLazyTensor(self.mats).matmul(self.vecs)
        actual = self.mats_copy.matmul(self.vecs_copy)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        grad_output = torch.randn(3, 4, 5, 2)
        res.backward(gradient=grad_output)
        actual.backward(gradient=grad_output)
        self.assertTrue(approx_equal(self.mats_copy.grad, self.mats.grad))
        self.assertTrue(approx_equal(self.vecs_copy.grad, self.vecs.grad))


if __name__ == "__main__":
    unittest.main()
