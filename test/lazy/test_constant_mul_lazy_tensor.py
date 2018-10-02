from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import ToeplitzLazyTensor
from gpytorch.utils.toeplitz import sym_toeplitz
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, BatchLazyTensorTestCase


class TestConstantMulLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        column = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        constant = 2.5
        return ToeplitzLazyTensor(column) * constant

    def evaluate_lazy_tensor(self, lazy_tensor):
        constant = lazy_tensor.constant
        column = lazy_tensor.base_lazy_tensor.column
        return sym_toeplitz(column) * constant


class TestConstantMulLazyTensorBatch(BatchLazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        column = torch.tensor([[5, 1, 2, 0]], dtype=torch.float).repeat(2, 1)
        column.requires_grad_(True)
        constant = torch.tensor([2.5, 1.])
        return ToeplitzLazyTensor(column) * constant

    def evaluate_lazy_tensor(self, lazy_tensor):
        constant = lazy_tensor.constant
        column = lazy_tensor.base_lazy_tensor.column
        return torch.cat([sym_toeplitz(column[0]).unsqueeze(0), sym_toeplitz(column[1]).unsqueeze(0)]) * constant.view(
            2, 1, 1
        )


if __name__ == "__main__":
    unittest.main()
