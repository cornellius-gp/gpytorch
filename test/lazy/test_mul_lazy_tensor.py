#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import MulLazyTensor, RootLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


def make_random_mat(size, rank, batch_size=None):
    if batch_size is None:
        res = torch.randn(size, rank)
        res.requires_grad_()
        return res
    else:
        res = torch.randn(batch_size, size, rank)
        res.requires_grad_()
        return res


class TestMulLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 10

    def create_lazy_tensor(self):
        mat1 = make_random_mat(6, 3)
        mat2 = make_random_mat(6, 3)
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2))
        return res.add_diag(torch.tensor(2.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        tensors = [lt.evaluate() for lt in lazy_tensor._lazy_tensor.lazy_tensors]
        res = tensors[0]
        for tensor in tensors[1:]:
            res = res * tensor
        res = res + diag_tensor
        return res


class TestMulLazyTensorMulti(LazyTensorTestCase, unittest.TestCase):
    seed = 10

    def test_quad_form_derivative(self):
        # MulLazyTensor creates non-leaf variables, so the default derivative
        # doesn't apply
        pass

    def create_lazy_tensor(self):
        mat1 = make_random_mat(30, 3)
        mat2 = make_random_mat(30, 3)
        mat3 = make_random_mat(30, 3)
        mat4 = make_random_mat(30, 3)
        mat5 = make_random_mat(30, 3)
        res = MulLazyTensor(
            RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3), RootLazyTensor(mat4), RootLazyTensor(mat5)
        )
        return res.add_diag(torch.tensor(1.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        tensors = [lt.evaluate() for lt in lazy_tensor._lazy_tensor.lazy_tensors]
        res = tensors[0]
        for tensor in tensors[1:]:
            res = res * tensor
        res = res + diag_tensor
        return res


class TestMulLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 2

    def create_lazy_tensor(self):
        mat1 = make_random_mat(6, rank=5, batch_size=2)
        mat2 = make_random_mat(6, rank=5, batch_size=2)
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2))
        return res.add_diag(torch.tensor(2.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        tensors = [lt.evaluate() for lt in lazy_tensor._lazy_tensor.lazy_tensors]
        res = tensors[0]
        for tensor in tensors[1:]:
            res = res * tensor
        res = res + diag_tensor
        return res


class TestMulLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 3

    def test_quad_form_derivative(self):
        # MulLazyTensor creates non-leaf variables, so the default derivative
        # doesn't apply
        pass

    def create_lazy_tensor(self):
        mat1 = make_random_mat(40, rank=5, batch_size=2)
        mat2 = make_random_mat(40, rank=5, batch_size=2)
        mat3 = make_random_mat(40, rank=5, batch_size=2)
        mat4 = make_random_mat(40, rank=5, batch_size=2)
        mat5 = make_random_mat(40, rank=5, batch_size=2)
        res = MulLazyTensor(
            RootLazyTensor(mat1), RootLazyTensor(mat2), RootLazyTensor(mat3), RootLazyTensor(mat4), RootLazyTensor(mat5)
        )
        return res.add_diag(torch.tensor(0.5))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        tensors = [lt.evaluate() for lt in lazy_tensor._lazy_tensor.lazy_tensors]
        res = tensors[0]
        for tensor in tensors[1:]:
            res = res * tensor
        res = res + diag_tensor
        return res


class TestMulLazyTensorWithConstantMul(LazyTensorTestCase, unittest.TestCase):
    seed = 2

    def test_quad_form_derivative(self):
        # MulLazyTensor creates non-leaf variables, so the default derivative
        # doesn't apply
        pass

    def create_lazy_tensor(self):
        mat1 = make_random_mat(20, rank=5, batch_size=2)
        mat2 = make_random_mat(20, rank=5, batch_size=2)
        constant = torch.tensor(4.0)
        res = MulLazyTensor(RootLazyTensor(mat1), RootLazyTensor(mat2))
        return res.mul(constant).add_diag(torch.tensor(2.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        tensors = [lt.evaluate() for lt in lazy_tensor._lazy_tensor.lazy_tensors]
        res = tensors[0]
        for tensor in tensors[1:]:
            res = res * tensor
        res = res + diag_tensor
        return res


if __name__ == "__main__":
    unittest.main()
