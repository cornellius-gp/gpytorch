#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestSumLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLazyTensor(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLazyTensor(c2)
        return t1 + t2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


class TestSumLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLazyTensor(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLazyTensor(c2)
        return t1 + t2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


class TestSumLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_lazy_tensor(self):
        mat1 = torch.randn(2, 3, 4, 4)
        lt1 = NonLazyTensor(mat1 @ mat1.transpose(-1, -2))
        mat2 = torch.randn(2, 3, 4, 4)
        lt2 = NonLazyTensor(mat2 @ mat2.transpose(-1, -2))
        return lt1 + lt2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


if __name__ == "__main__":
    unittest.main()
