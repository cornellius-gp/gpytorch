#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import LowRankRootLazyTensor
from gpytorch.test.lazy_tensor_test_case import RectangularLazyTensorTestCase


class TestLowRankRootLazyTensor(RectangularLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        root = torch.randn(3, 1, requires_grad=True)
        return LowRankRootLazyTensor(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


class TestLowRankRootLazyTensorBatch(TestLowRankRootLazyTensor):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, 2)
        return LowRankRootLazyTensor(root)


class TestLowRankRootLazyTensorMultiBatch(TestLowRankRootLazyTensor):
    seed = 1
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(4, 3, 5, 2)
        return LowRankRootLazyTensor(root)


if __name__ == "__main__":
    unittest.main()
