#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import MatmulLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, RectangularLazyTensorTestCase


class TestMatmulLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 1

    def create_lazy_tensor(self):
        lhs = torch.randn(5, 6, requires_grad=True)
        rhs = lhs.clone().detach().transpose(-1, -2)
        covar = MatmulLazyTensor(lhs, rhs)
        return covar

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.left_lazy_tensor.tensor.matmul(lazy_tensor.right_lazy_tensor.tensor)


class TestMatmulLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 3

    def create_lazy_tensor(self):
        lhs = torch.randn(5, 5, 6, requires_grad=True)
        rhs = lhs.clone().detach().transpose(-1, -2)
        covar = MatmulLazyTensor(lhs, rhs)
        return covar

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.left_lazy_tensor.tensor.matmul(lazy_tensor.right_lazy_tensor.tensor)


class TestMatmulLazyTensorRectangular(RectangularLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        lhs = torch.randn(5, 3, requires_grad=True)
        rhs = torch.randn(3, 6, requires_grad=True)
        covar = MatmulLazyTensor(lhs, rhs)
        return covar

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.left_lazy_tensor.tensor.matmul(lazy_tensor.right_lazy_tensor.tensor)


class TestMatmulLazyTensorRectangularBatch(RectangularLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        lhs = torch.randn(3, 5, 3, requires_grad=True)
        rhs = torch.randn(3, 3, 6, requires_grad=True)
        covar = MatmulLazyTensor(lhs, rhs)
        return covar

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.left_lazy_tensor.tensor.matmul(lazy_tensor.right_lazy_tensor.tensor)


if __name__ == "__main__":
    unittest.main()
