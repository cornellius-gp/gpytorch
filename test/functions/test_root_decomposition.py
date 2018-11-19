#!/usr/bin/env python3

import os
import torch
import random
import unittest
from gpytorch.lazy import NonLazyTensor
from test._utils import approx_equal


class TestRootDecomposition(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        mat = torch.randn(4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4))
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_clone = mat.detach().clone().requires_grad_(True)

    def test_root_decomposition(self):
        # Forward
        root = NonLazyTensor(self.mat).root_decomposition().root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        self.assertTrue(approx_equal(res, self.mat))

        # Backward
        res.trace().backward()
        self.mat_clone.trace().backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))

    def test_root_inv_decomposition(self):
        # Forward
        probe_vectors = torch.randn(4, 5)
        test_vectors = torch.randn(4, 5)
        root = NonLazyTensor(self.mat).root_inv_decomposition(probe_vectors, test_vectors).root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        actual = self.mat_clone.inverse()
        self.assertTrue(approx_equal(res, actual))

        # Backward
        res.trace().backward()
        actual.trace().backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))


class TestRootDecompositionBatch(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        mat = torch.randn(3, 4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4).unsqueeze_(0))
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_clone = mat.detach().clone().requires_grad_(True)

    def test_root_decomposition(self):
        # Forward
        root = NonLazyTensor(self.mat).root_decomposition().root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        self.assertTrue(approx_equal(res, self.mat))

        # Backward
        sum([mat.trace() for mat in res]).backward()
        sum([mat.trace() for mat in self.mat_clone]).backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))

    def test_root_inv_decomposition(self):
        # Forward
        probe_vectors = torch.randn(3, 4, 5)
        test_vectors = torch.randn(3, 4, 5)
        root = NonLazyTensor(self.mat).root_inv_decomposition(probe_vectors, test_vectors).root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        actual = torch.cat([mat.inverse().unsqueeze(0) for mat in self.mat_clone])
        self.assertTrue(approx_equal(res, actual))

        # Backward
        sum([mat.trace() for mat in res]).backward()
        sum([mat.trace() for mat in actual]).backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))


class TestRootDecompositionMultiBatch(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def setUp(self):
        if os.getenv("unlock_seed") is None or os.getenv("unlock_seed").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

        mat = torch.randn(2, 3, 4, 4)
        mat = mat @ mat.transpose(-1, -2)
        mat.div_(5).add_(torch.eye(4).view(1, 1, 4, 4))
        self.mat = mat.detach().clone().requires_grad_(True)
        self.mat_clone = mat.detach().clone().requires_grad_(True)

    def test_root_decomposition(self):
        # Forward
        root = NonLazyTensor(self.mat).root_decomposition().root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        self.assertTrue(approx_equal(res, self.mat))

        # Backward
        sum([mat.trace() for mat in res.view(-1, *self.mat.shape[-2:])]).backward()
        sum([mat.trace() for mat in self.mat_clone.view(-1, *self.mat.shape[-2:])]).backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))

    def test_root_inv_decomposition(self):
        # Forward
        probe_vectors = torch.randn(2, 3, 4, 5)
        test_vectors = torch.randn(2, 3, 4, 5)
        root = NonLazyTensor(self.mat).root_inv_decomposition(probe_vectors, test_vectors).root.evaluate()
        res = root.matmul(root.transpose(-1, -2))
        flattened_mats = self.mat_clone.view(-1, *self.mat_clone.shape[-2:])
        actual = torch.cat([mat.inverse().unsqueeze(0) for mat in flattened_mats]).view_as(self.mat_clone)
        self.assertTrue(approx_equal(res, actual))

        # Backward
        sum([mat.trace() for mat in res.view(-1, *self.mat.shape[-2:])]).backward()
        sum([mat.trace() for mat in actual.view(-1, *self.mat.shape[-2:])]).backward()
        self.assertTrue(approx_equal(self.mat.grad, self.mat_clone.grad))


if __name__ == "__main__":
    unittest.main()
