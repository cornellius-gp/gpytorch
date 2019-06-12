#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor, BatchRepeatLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestBatchRepeatLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor([4, 0.1, 0.05, 0.01, 0.0], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLazyTensor(ToeplitzLazyTensor(toeplitz_column), torch.Size((3,)))

    def evaluate_lazy_tensor(self, lazy_tensor):
        evaluated = lazy_tensor.base_lazy_tensor.evaluate()
        return evaluated.repeat(*lazy_tensor.batch_repeat, 1, 1)


class TestBatchRepeatLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor([[4, 0, 0, 1], [3, 0, -0.5, -1]], dtype=torch.float)
        toeplitz_column.detach_()
        return BatchRepeatLazyTensor(ToeplitzLazyTensor(toeplitz_column), torch.Size((3,)))
        return BatchRepeatLazyTensor(ToeplitzLazyTensor(toeplitz_column), torch.Size((3,)))

    def evaluate_lazy_tensor(self, lazy_tensor):
        evaluated = lazy_tensor.base_lazy_tensor.evaluate()
        return evaluated.repeat(*lazy_tensor.batch_repeat, 1, 1)


class TestBatchRepeatLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor(
            [[[4, 0, 0, 1], [3, 0, -0.5, -1]], [[2, 0.1, 0.01, 0.0], [3, 0, -0.1, -2]]], dtype=torch.float
        )
        toeplitz_column.detach_()
        return BatchRepeatLazyTensor(ToeplitzLazyTensor(toeplitz_column), torch.Size((2, 3, 1, 4)))

    def evaluate_lazy_tensor(self, lazy_tensor):
        evaluated = lazy_tensor.base_lazy_tensor.evaluate()
        return evaluated.repeat(*lazy_tensor.batch_repeat, 1, 1)


if __name__ == "__main__":
    unittest.main()
