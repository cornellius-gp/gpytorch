#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor
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


if __name__ == "__main__":
    unittest.main()
