from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import unittest
from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase, BatchLazyTensorTestCase
from test.lazy._lazy_tensor_test_case import RectangularLazyTensorTestCase, RectangularBatchLazyTensorTestCase


def kron(a, b):
    res = []
    if b.ndimension() == 2:
        for i in range(b.size(0)):
            row_res = []
            for j in range(b.size(1)):
                row_res.append(a * b[i, j])
            res.append(torch.cat(row_res, 1))
        return torch.cat(res, 0)
    else:
        for i in range(b.size(1)):
            row_res = []
            for j in range(b.size(2)):
                row_res.append(a * b[:, i, j].unsqueeze(1).unsqueeze(2))
            res.append(torch.cat(row_res, 2))
        return torch.cat(res, 1)


class TestKroneckerProductLazyTensor(LazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0, 1, 0], [0, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = kron(lazy_tensor.lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].tensor)
        res = kron(res, lazy_tensor.lazy_tensors[2].tensor)
        return res


class TestKroneckerProductLazyTensorBatch(BatchLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor([[4, 0, 1, 0], [0, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float).repeat(3, 1, 1)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = kron(lazy_tensor.lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].tensor)
        res = kron(res, lazy_tensor.lazy_tensors[2].tensor)
        return res


class TestKroneckerProductLazyTensorRectangular(RectangularLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        a = torch.randn(2, 3, requires_grad=True)
        b = torch.randn(5, 2, requires_grad=True)
        c = torch.randn(6, 4, requires_grad=True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = kron(lazy_tensor.lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].tensor)
        res = kron(res, lazy_tensor.lazy_tensors[2].tensor)
        return res


class TestKroneckerProductLazyTensorRectangularBatch(RectangularBatchLazyTensorTestCase, unittest.TestCase):
    def create_lazy_tensor(self):
        a = torch.randn(4, 2, 3, requires_grad=True)
        b = torch.randn(4, 5, 2, requires_grad=True)
        c = torch.randn(4, 6, 4, requires_grad=True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = kron(lazy_tensor.lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].tensor)
        res = kron(res, lazy_tensor.lazy_tensors[2].tensor)
        return res


if __name__ == "__main__":
    unittest.main()
