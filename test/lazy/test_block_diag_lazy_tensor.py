#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import BlockDiagLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestBlockDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        return BlockDiagLazyTensor(NonLazyTensor(blocks))

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            actual[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = blocks[i]
        return actual


class TestBlockDiagLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        return BlockDiagLazyTensor(NonLazyTensor(blocks), block_dim=2)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(2, 24, 24)
        for i in range(2):
            for j in range(6):
                actual[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = blocks[i, j]
        return actual


class TestBlockDiagLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 6, 5, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        blocks.detach_()
        return BlockDiagLazyTensor(NonLazyTensor(blocks), block_dim=1)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(2, 5, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(5):
                    actual[i, k, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = blocks[i, k, j]
        return actual


if __name__ == "__main__":
    unittest.main()
