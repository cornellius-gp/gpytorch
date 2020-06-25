#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import DiagLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestDiagLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_cg = False
    should_call_lanczos = False

    def create_lazy_tensor(self):
        diag = torch.tensor([1.0, 2.0, 4.0, 5.0, 3.0], requires_grad=True)
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return diag.diag()


class TestDiagLazyTensorBatch(TestDiagLazyTensor):
    seed = 0

    def create_lazy_tensor(self):
        diag = torch.tensor(
            [[1.0, 2.0, 4.0, 2.0, 3.0], [2.0, 1.0, 2.0, 1.0, 4.0], [1.0, 2.0, 2.0, 3.0, 4.0]], requires_grad=True
        )
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        return torch.cat([diag[i].diag().unsqueeze(0) for i in range(3)])


class TestDiagLazyTensorMultiBatch(TestDiagLazyTensor):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = True
    skip_slq_tests = True

    def create_lazy_tensor(self):
        diag = torch.randn(6, 3, 5).pow_(2)
        diag.requires_grad_(True)
        return DiagLazyTensor(diag)

    def evaluate_lazy_tensor(self, lazy_tensor):
        diag = lazy_tensor._diag
        flattened_diag = diag.view(-1, diag.size(-1))
        res = torch.cat([flattened_diag[i].diag().unsqueeze(0) for i in range(18)])
        return res.view(6, 3, 5, 5)


if __name__ == "__main__":
    unittest.main()
