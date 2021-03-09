#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import CholLazyTensor, TriangularLazyTensor
from gpytorch.test.lazy_tensor_test_case import LazyTensorTestCase


class TestCholLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True
    should_call_cg = False
    should_call_lanczos = False
    should_call_lanczos_diagonalization = True

    def create_lazy_tensor(self):
        chol = torch.tensor(
            [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
            dtype=torch.float,
            requires_grad=True,
        )
        return CholLazyTensor(TriangularLazyTensor(chol))

    def evaluate_lazy_tensor(self, lazy_tensor):
        chol = lazy_tensor.root.evaluate()
        return chol.matmul(chol.transpose(-1, -2))


class TestCholLazyTensorBatch(TestCholLazyTensor):
    seed = 0

    def create_lazy_tensor(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
        )
        chol.add_(torch.eye(5).unsqueeze(0))
        chol.requires_grad_(True)
        return CholLazyTensor(TriangularLazyTensor(chol))


class TestCholLazyTensorMultiBatch(TestCholLazyTensor):
    seed = 0
    # Because these LTs are large, we'll skil the big tests
    should_test_sample = False
    skip_slq_tests = True

    def create_lazy_tensor(self):
        chol = torch.tensor(
            [
                [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
                [[2, 0, 0, 0, 0], [3, 1, 0, 0, 0], [-2, 3, 2, 0, 0], [-2, 1, -1, 3, 0], [-4, -4, 5, 2, 3]],
            ],
            dtype=torch.float,
        )
        chol = chol.repeat(3, 1, 1, 1)
        chol[1].mul_(2)
        chol[2].mul_(0.5)
        chol.add_(torch.eye(5).unsqueeze_(0).unsqueeze_(0))
        chol.requires_grad_(True)
        return CholLazyTensor(TriangularLazyTensor(chol))


if __name__ == "__main__":
    unittest.main()
