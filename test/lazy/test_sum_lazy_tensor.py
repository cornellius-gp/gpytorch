#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import ToeplitzLazyTensor, lazify
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestSumLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        c1 = torch.tensor([5, 1, 2, 0], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLazyTensor(c1)
        c2 = torch.tensor([6, 0, 1, -1], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLazyTensor(c2)
        return t1 + t2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


class TestSumLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0

    def create_lazy_tensor(self):
        c1 = torch.tensor([[2, 0.5, 0, 0], [5, 1, 2, 0]], dtype=torch.float, requires_grad=True)
        t1 = ToeplitzLazyTensor(c1)
        c2 = torch.tensor([[2, 0.5, 0, 0], [6, 0, 1, -1]], dtype=torch.float, requires_grad=True)
        t2 = ToeplitzLazyTensor(c2)
        return t1 + t2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


class TestSumLazyTensorMultiBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    skip_slq_tests = True

    def create_lazy_tensor(self):
        c1 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [5, 1, 2, 0]]], dtype=torch.float, requires_grad=True,
        )
        t1 = ToeplitzLazyTensor(c1)
        c2 = torch.tensor(
            [[[2, 0.5, 0, 0], [5, 1, 2, 0]], [[2, 0.5, 0, 0], [6, 0, 1, -1]]], dtype=torch.float, requires_grad=True,
        )
        t2 = ToeplitzLazyTensor(c2)
        return t1 + t2

    def evaluate_lazy_tensor(self, lazy_tensor):
        tensors = [lt.evaluate() for lt in lazy_tensor.lazy_tensors]
        return sum(tensors)


class TestSumLazyTensorBroadcasting(unittest.TestCase):
    def test_broadcast_same_shape(self):
        test1 = lazify(torch.randn(30, 30))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_tensor_shape(self):
        test1 = lazify(torch.randn(30, 30))

        test2 = torch.randn(30, 1)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)

    def test_broadcast_lazy_shape(self):
        test1 = lazify(torch.randn(30, 1))

        test2 = torch.randn(30, 30)
        res = test1 + test2
        final_res = res + test2

        torch_res = res.evaluate() + test2

        self.assertEqual(final_res.shape, torch_res.shape)
        self.assertEqual((final_res.evaluate() - torch_res).sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
