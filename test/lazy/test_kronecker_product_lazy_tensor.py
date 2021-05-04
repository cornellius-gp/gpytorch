#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import DiagLazyTensor, KroneckerProductDiagLazyTensor, KroneckerProductLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase, RectangularLazyTensorTestCase

from .test_diag_lazy_tensor import TestDiagLazyTensor


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


def kron_diag(*lts):
    """Compute diagonal of a KroneckerProductLazyTensor from the diagonals of the constituiting tensors"""
    lead_diag = lts[0].diag()
    if len(lts) == 1:  # base case:
        return lead_diag
    trail_diag = kron_diag(*lts[1:])
    diag = lead_diag.unsqueeze(-2) * trail_diag.unsqueeze(-1)
    return diag.transpose(-1, -2).reshape(*diag.shape[:-2], -1)


class TestKroneckerProductLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1, 0], [0.5, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res = kron(lazy_tensor.lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].tensor)
        res = kron(res, lazy_tensor.lazy_tensors[2].tensor)
        return res


class TestKroneckerProductDiagLazyTensor(TestDiagLazyTensor):
    should_call_lanczos_diagonalization = False

    def create_lazy_tensor(self):
        a = torch.tensor([4.0, 1.0, 2.0], dtype=torch.float)
        b = torch.tensor([3.0, 1.3], dtype=torch.float)
        c = torch.tensor([1.75, 1.95, 1.05, 0.25], dtype=torch.float)
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductDiagLazyTensor(DiagLazyTensor(a), DiagLazyTensor(b), DiagLazyTensor(c))
        return kp_lazy_tensor

    def evaluate_lazy_tensor(self, lazy_tensor):
        res_diag = kron_diag(*lazy_tensor.lazy_tensors)
        return torch.diag_embed(res_diag)


class TestKroneckerProductLazyTensorBatch(TestKroneckerProductLazyTensor):
    seed = 0
    should_call_lanczos = True
    should_call_lanczos_diagonalization = False

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float).repeat(3, 1, 1)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float).repeat(3, 1, 1)
        c = torch.tensor([[4, 0.1, 1, 0], [0.1, 4, -1, 0], [1, -1, 3, 0], [0, 0, 0, 4]], dtype=torch.float).repeat(
            3, 1, 1
        )
        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor


class TestKroneckerProductLazyTensorRectangular(RectangularLazyTensorTestCase, unittest.TestCase):
    seed = 0

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


class TestKroneckerProductLazyTensorRectangularMultiBatch(TestKroneckerProductLazyTensorRectangular):
    seed = 0

    def create_lazy_tensor(self):
        a = torch.randn(3, 4, 2, 3, requires_grad=True)
        b = torch.randn(3, 4, 5, 2, requires_grad=True)
        c = torch.randn(3, 4, 6, 4, requires_grad=True)
        kp_lazy_tensor = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b), NonLazyTensor(c))
        return kp_lazy_tensor


if __name__ == "__main__":
    unittest.main()
