from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import DiagLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, BatchLazyTensorTestCase


class TestDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0], requires_grad=True)
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return diag.diag()


class TestDiagLazyTensorBatch(BatchLazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
        )
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return torch.cat([diag[i].diag().unsqueeze(0) for i in range(3)])


if __name__ == "__main__":
    unittest.main()
