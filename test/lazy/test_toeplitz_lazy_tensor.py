#!/usr/bin/env python3

import unittest

import torch

import gpytorch.utils.toeplitz as toeplitz
from gpytorch.lazy import ToeplitzLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestToeplitzLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 1

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor([4, 0.5, 0, 1], dtype=torch.float, requires_grad=True)
        return ToeplitzLazyTensor(toeplitz_column)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return toeplitz.sym_toeplitz(lazy_tensor.column)


class TestToeplitzLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float, requires_grad=True)
        return ToeplitzLazyTensor(toeplitz_column)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(lazy_tensor.column[0]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[1]).unsqueeze(0),
            ]
        )


class TestToeplitzLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        toeplitz_column = torch.tensor([[2, -1, 0.5, 0.25], [4, 0.5, 0, 1]], dtype=torch.float)
        toeplitz_column = toeplitz_column.repeat(3, 1, 1)
        toeplitz_column.requires_grad_(True)
        return ToeplitzLazyTensor(toeplitz_column)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return torch.cat(
            [
                toeplitz.sym_toeplitz(lazy_tensor.column[0, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[0, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[1, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[1, 1]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[2, 0]).unsqueeze(0),
                toeplitz.sym_toeplitz(lazy_tensor.column[2, 1]).unsqueeze(0),
            ]
        ).view(3, 2, 4, 4)


if __name__ == "__main__":
    unittest.main()
