#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import KroneckerProductLazyTensor, NonLazyTensor, SumKroneckerLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


def kron(a, b):
    res = []
    for i in range(a.size(-2)):
        row_res = []
        for j in range(a.size(-1)):
            row_res.append(b * a[..., i, j].unsqueeze(-1).unsqueeze(-2))
        res.append(torch.cat(row_res, -1))
    return torch.cat(res, -2)


class TestSumKroneckerLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_call_lanczos = True
    should_call_cg = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        a = torch.tensor([[4, 0, 2], [0, 3, -1], [2, -1, 3]], dtype=torch.float)
        b = torch.tensor([[2, 1], [1, 2]], dtype=torch.float)
        c = torch.tensor([[4, 0.5, 1], [0.5, 4, -1], [1, -1, 3]], dtype=torch.float)
        d = torch.tensor([[1.2, 0.75], [0.75, 1.2]], dtype=torch.float)

        a.requires_grad_(True)
        b.requires_grad_(True)
        c.requires_grad_(True)
        d.requires_grad_(True)
        kp_lt_1 = KroneckerProductLazyTensor(NonLazyTensor(a), NonLazyTensor(b))
        kp_lt_2 = KroneckerProductLazyTensor(NonLazyTensor(c), NonLazyTensor(d))

        return SumKroneckerLazyTensor(kp_lt_1, kp_lt_2)

    def evaluate_lazy_tensor(self, lazy_tensor):
        res1 = kron(
            lazy_tensor.lazy_tensors[0].lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[0].lazy_tensors[1].tensor
        )
        res2 = kron(
            lazy_tensor.lazy_tensors[1].lazy_tensors[0].tensor, lazy_tensor.lazy_tensors[1].lazy_tensors[1].tensor
        )
        return res1 + res2
