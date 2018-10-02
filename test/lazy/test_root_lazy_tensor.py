from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import RootLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, BatchLazyTensorTestCase


class TestRootLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        root = torch.randn(5, 3, requires_grad=True)
        return RootLazyTensor(root).add_diag(torch.tensor(1.0))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag_tensor = lazy_tensor._diag_tensor.evaluate()
        root = lazy_tensor._lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        res = res + diag_tensor
        return res


class TestRootLazyTensorBatch(BatchLazyTensorTestCase, unittest.TestCase):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(3, 5, 5)
        root.add_(torch.eye(5).unsqueeze(0))
        root.requires_grad_(True)
        return RootLazyTensor(root)

    def evaluate_lazy_tensor(self, lazy_tensor):
        root = lazy_tensor.root.tensor
        res = root.matmul(root.transpose(-1, -2))
        return res


if __name__ == "__main__":
    unittest.main()
