#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import NonLazyTensor, SumBatchLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestSumBatchLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 6
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(12, 4, 4)
        blocks = blocks.transpose(-1, -2).matmul(blocks)
        blocks.requires_grad_(True)
        return SumBatchLazyTensor(NonLazyTensor(blocks))

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        return blocks.sum(0)


class TestSumBatchLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 6
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.transpose(-1, -2).matmul(blocks)
        blocks.requires_grad_(True)
        return SumBatchLazyTensor(NonLazyTensor(blocks))

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        return blocks.view(2, 6, 4, 4).sum(1)


class TestSumBatchLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 6
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 3, 6, 4, 4)
        blocks = blocks.transpose(-1, -2).matmul(blocks)
        blocks.detach_()
        return SumBatchLazyTensor(NonLazyTensor(blocks), block_dim=1)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        return blocks.sum(-3)


if __name__ == "__main__":
    unittest.main()
