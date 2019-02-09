#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import NonLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestAddedDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0], requires_grad=True)
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + diag.diag()


class TestAddedDiagLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 4
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(3, 5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
        )
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + torch.diag_embed(diag, dim1=-2, dim2=-1)


class TestAddedDiagLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 4
    should_test_sample = True

    def create_lazy_tensor(self):
        tensor = torch.randn(4, 3, 5, 5)
        tensor = tensor.transpose(-1, -2).matmul(tensor).detach()
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
        ).repeat(4, 1, 1).detach()
        return AddedDiagLazyTensor(NonLazyTensor(tensor), DiagLazyTensor(diag))

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag_tensor._diag
        tensor = lazy_tensor._lazy_tensor.tensor
        return tensor + torch.diag_embed(diag, dim1=-2, dim2=-1)


if __name__ == "__main__":
    unittest.main()
