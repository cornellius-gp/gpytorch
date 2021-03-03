#!/usr/bin/env python3

import unittest

import torch

from gpytorch.lazy import NonLazyTensor
from gpytorch.test.base_test_case import BaseTestCase


class TestRootDecomposition(BaseTestCase, unittest.TestCase):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4))
        return mat

    def test_root_decomposition(self):
        mat = self._create_mat().detach().requires_grad_(True)
        mat_clone = mat.detach().clone().requires_grad_(True)

        # Forward
        root = NonLazyTensor(mat).root_decomposition().root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        self.assertAllClose(res, mat)

        # Backward
        sum([mat.trace() for mat in res.view(-1, mat.size(-2), mat.size(-1))]).backward()
        sum([mat.trace() for mat in mat_clone.view(-1, mat.size(-2), mat.size(-1))]).backward()
        self.assertAllClose(mat.grad, mat_clone.grad)

    def test_root_inv_decomposition(self):
        mat = self._create_mat().detach().requires_grad_(True)
        mat_clone = mat.detach().clone().requires_grad_(True)

        # Forward
        probe_vectors = torch.randn(*mat.shape[:-2], 4, 5)
        test_vectors = torch.randn(*mat.shape[:-2], 4, 5)
        root = (
            NonLazyTensor(mat)
            .root_inv_decomposition(method="lanczos", initial_vectors=probe_vectors, test_vectors=test_vectors)
            .root.evaluate()
        )
        res = root.matmul(root.transpose(-1, -2))
        actual = mat_clone.inverse()
        self.assertAllClose(res, actual)

        # Backward
        sum([mat.trace() for mat in res.view(-1, mat.size(-2), mat.size(-1))]).backward()
        sum([mat.trace() for mat in actual.view(-1, mat.size(-2), mat.size(-1))]).backward()
        self.assertAllClose(mat.grad, mat_clone.grad)


class TestRootDecompositionBatch(TestRootDecomposition):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(3, 4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4).unsqueeze_(0))
        return mat


class TestRootDecompositionMultiBatch(TestRootDecomposition):
    seed = 0

    def _create_mat(self):
        mat = torch.randn(2, 3, 4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4).unsqueeze_(0))
        return mat


if __name__ == "__main__":
    unittest.main()
