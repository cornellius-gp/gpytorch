#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import RootLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestRootLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_lanczos = False
    should_call_lanczos_diagonalization = True

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, requires_grad=True)
        return RootLazyTensor(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestRootLazyTensorBatch(TestRootLazyTensor):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLazyTensor(root)


class TestRootLazyTensorMultiBatch(TestRootLazyTensor):
    seed = 2
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(2, 3, 5, 5) + torch.eye(5)
        root.requires_grad_(True)
        return RootLazyTensor(root)


if __name__ == "__main__":
    unittest.main()
