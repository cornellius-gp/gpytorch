#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import BlockDiagLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestBlockDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        blocks.requires_grad_(True)
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
        blocks = torch.randn(8, 4, 4)
        blocks = blocks.matmul(blocks.transpose(-1, -2))
        blocks.add_(torch.eye(4, 4).unsqueeze_(0))
        blocks.requires_grad_(True)
        return BlockDiagLazyTensor(NonLazyTensor(blocks), num_blocks=4)

    def evaluate_lazy_tensor(self, lazy_tensor):
        blocks = lazy_tensor.base_lazy_tensor.tensor
        actual = torch.zeros(2, 16, 16)
        for i in range(2):
            for j in range(4):
                actual[i, j * 4 : (j + 1) * 4, j * 4 : (j + 1) * 4] = blocks[i * 4 + j]
        return actual


if __name__ == "__main__":
    unittest.main()
