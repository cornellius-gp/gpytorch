#!/usr/bin/env python3

import torch
import unittest
from gpytorch.lazy import CholLazyTensor
from test.lazy._lazy_tensor_test_case import LazyTensorTestCase


class TestCholLazyTensor(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

    def create_lazy_tensor(self):
        chol = torch.tensor(
            [[3, 0, 0, 0, 0], [-1, 2, 0, 0, 0], [1, 4, 1, 0, 0], [0, 2, 3, 2, 0], [-4, -2, 1, 3, 4]],
            dtype=torch.float,
            requires_grad=True,
        )
        return CholLazyTensor(chol)

    def evaluate_lazy_tensor(self, lazy_tensor):
        chol = lazy_tensor.root.evaluate()
        return chol.matmul(chol.transpose(-1, -2))

    def test_inv_matmul_vec(self):
        # We're skipping this test, since backward passes aren't defined in torch for this
        pass

    def test_inv_matmul_matrix(self):
        # We're skipping this test, since backward passes aren't defined in torch for this
        pass


class TestCholLazyTensorBatch(LazyTensorTestCase, unittest.TestCase):
    seed = 0
    should_test_sample = True

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
        return CholLazyTensor(chol)

    def evaluate_lazy_tensor(self, lazy_tensor):
        chol = lazy_tensor.root.evaluate()
        res = chol.matmul(chol.transpose(-1, -2))
        return res


if __name__ == "__main__":
    unittest.main()
