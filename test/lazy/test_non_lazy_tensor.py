#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestNonLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLazyTensor(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.tensor


class TestNonLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        mat = torch.randn(3, 5, 6)
        mat = mat.matmul(mat.transpose(-1, -2))
        mat.requires_grad_(True)
        return NonLazyTensor(mat)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return lazy_tensor.tensor
