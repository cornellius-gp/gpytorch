#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import SumBatchLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


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
        blocks = torch.randn(12, 4, 4)
        blocks = blocks.transpose(-1, -2).matmul(blocks)
        blocks.requires_grad_(True)
        return SumBatchLazyTensor(NonLazyTensor(blocks), num_blocks=6)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        return blocks.view(2, 6, 4, 4).sum(1)


if __name__ == "__main__":
    unittest.main()
