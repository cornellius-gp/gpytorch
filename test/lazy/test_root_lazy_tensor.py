#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import RootLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestRootLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, requires_grad=True)
        return RootLazyTensor(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestRootLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, 5)
        root.add_(torch.eye(5).unsqueeze(0))
        root.requires_grad_(True)
        return RootLazyTensor(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


if __name__ == "__main__":
    unittest.main()
