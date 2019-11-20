#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import CatLazyTensor, NonLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestCatLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 1

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


class TestCatLazyTensorColumn(LazyTensorTestCase, unittest.TestCase):
    seed = 1

    def create_lazy_tensor(self):
        root = torch.randn(6, 7)
        self.psd_mat = root.matmul(root.t())

        slice1_mat = self.psd_mat[:, :2].requires_grad_()
        slice2_mat = self.psd_mat[:, 2:4].requires_grad_()
        slice3_mat = self.psd_mat[:, 4:6].requires_grad_()

        slice1 = NonLazyTensor(slice1_mat)
        slice2 = NonLazyTensor(slice2_mat)
        slice3 = NonLazyTensor(slice3_mat)

        return CatLazyTensor(slice1, slice2, slice3, dim=-1)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return self.psd_mat.detach().clone().requires_grad_()


class TestCatLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

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


class TestCatLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(4, 3, 6, 7)
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


class TestCatLazyTensorBatchCat(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_lazy_tensor(self):
        root = torch.randn(5, 3, 6, 7)
        self.psd_mat = root.matmul(root.transpose(-2, -1))

        slice1_mat = self.psd_mat[:2, ...].requires_grad_()
        slice2_mat = self.psd_mat[2:3, ...].requires_grad_()
        slice3_mat = self.psd_mat[3:, ...].requires_grad_()

        slice1 = NonLazyTensor(slice1_mat)
        slice2 = NonLazyTensor(slice2_mat)
        slice3 = NonLazyTensor(slice3_mat)

        return CatLazyTensor(slice1, slice2, slice3, dim=0)

    def evaluate_lazy_tensor(self, lazy_tensor):
        return self.psd_mat.detach().clone().requires_grad_()


if __name__ == "__main__":
    unittest.main()
