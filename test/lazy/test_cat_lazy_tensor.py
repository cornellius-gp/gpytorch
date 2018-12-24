#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import CatLazyTensor, NonLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestCatLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    no_broadcast_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:2, :].requires_grad_()
        slice2_mat = self.psd_mat[2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[4:6, :].requires_grad_()

        slice1 = NonLazyTensor(slice1_mat)
        slice2 = NonLazyTensor(slice2_mat)
        slice3 = NonLazyTensor(slice3_mat)

        return CatLazyTensor(slice1, slice2, slice3, dim=-2)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    no_broadcast_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(3, 6, 7)
        self.psd_mat = root.matmul(root.transpose(-2, -1))

        slice1_mat = self.psd_mat[..., :2, :].requires_grad_()
        slice2_mat = self.psd_mat[..., 2:4, :].requires_grad_()
        slice3_mat = self.psd_mat[..., 4:6, :].requires_grad_()

        slice1 = NonLazyTensor(slice1_mat)
        slice2 = NonLazyTensor(slice2_mat)
        slice3 = NonLazyTensor(slice3_mat)

        return CatLazyTensor(slice1, slice2, slice3, dim=-2)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return self.psd_mat.detach().clone().requires_grad_()


if __name__ == "__main__":
    unittest.main()
