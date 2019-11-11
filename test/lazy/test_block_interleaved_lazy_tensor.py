#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import BlockInterleavedLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestBlockInterleavedLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        return BlockInterleavedLazyTensor(NonLazyTensor(blocks))

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(32, 32)
        for i in range(8):
            for j in range(4):
                for k in range(4):
                    actual[j * 8 + i, k * 8 + i] = blocks[i, j, k]
        return actual


class TestBlockInterleavedLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 6, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        return BlockInterleavedLazyTensor(NonLazyTensor(blocks), block_dim=2)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(2, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(4):
                    for l in range(4):
                        actual[i, k * 6 + j, l * 6 + j] = blocks[i, j, k, l]
        return actual


class TestBlockInterleavedLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        blocks = torch.randn(2, 6, 5, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4))
        blocks.detach_()
        return BlockInterleavedLazyTensor(NonLazyTensor(blocks), block_dim=1)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(2, 5, 24, 24)
        for i in range(2):
            for j in range(6):
                for k in range(5):
                    for l in range(4):
                        for m in range(4):
                            actual[i, k, l * 6 + j, m * 6 + j] = blocks[i, k, j, l, m]
        return actual


if __name__ == "__main__":
    unittest.main()
